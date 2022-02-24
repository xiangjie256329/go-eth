// Copyright 2017 The go-ethereum Authors
// This file is part of the go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package core

import (
	"context"
	"encoding/binary"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/rawdb"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/ethdb"
	"github.com/ethereum/go-ethereum/event"
	"github.com/ethereum/go-ethereum/log"
)

// ChainIndexerBackend defines the methods needed to process chain segments in
// the background and write the segment results into the database. These can be
// used to create filter blooms or CHTs.
// ChainIndexerBackend定义了处理区块链片段的方法，并把处理结果写入数据库。 这些可以用来创建布隆过滤器或者CHTs.
// BloomIndexer 其实就是实现了这个接口 ChainIndexerBackend 这里的CHTs不知道是什么东西。
type ChainIndexerBackend interface {
	// Reset initiates the processing of a new chain segment, potentially terminating
	// any partially completed operations (in case of a reorg).
	// Reset 方法用来初始化一个新的区块链片段，可能会终止任何没有完成的操作。
	Reset(ctx context.Context, section uint64, prevHead common.Hash) error

	// Process crunches through the next header in the chain segment. The caller
	// will ensure a sequential order of headers.
	// 对区块链片段中的下一个区块头进行处理。 调用者将确保区块头的连续顺序。
	Process(ctx context.Context, header *types.Header) error

	// Commit finalizes the section metadata and stores it into the database.
	// 完成区块链片段的元数据并将其存储到数据库中。
	Commit() error

	// Prune deletes the chain index older than the given threshold.
	// Prune 删除早于给定阈值的链索引。用于裁剪
	Prune(threshold uint64) error
}

// ChainIndexerChain interface is used for connecting the indexer to a blockchain
type ChainIndexerChain interface {
	// CurrentHeader retrieves the latest locally known header.
	CurrentHeader() *types.Header

	// SubscribeChainHeadEvent subscribes to new head header notifications.
	SubscribeChainHeadEvent(ch chan<- ChainHeadEvent) event.Subscription
}

// ChainIndexer does a post-processing job for equally sized sections of the
// canonical chain (like BlooomBits and CHT structures). A ChainIndexer is
// connected to the blockchain through the event system by starting a
// ChainHeadEventLoop in a goroutine.
// ChainIndexer 对区块链进行 大小相等的片段 进行处。 ChainIndexer在ChainEventLoop方法中通过事件系统与区块链通信，
// Further child ChainIndexers can be added which use the output of the parent
// section indexer. These child indexers receive new head notifications only
// after an entire section has been finished or in case of rollbacks that might
// affect already finished sections.
// 更远可以添加使用父section索引器的输出的更多子链式索引器。 这些子链式索引器只有在整个部分完成后或在可能影响已完成部分的回滚的情况下才接收新的头部通知。
type ChainIndexer struct {
	chainDb  ethdb.Database      // Chain database to index the data from	区块链所在的数据库
	indexDb  ethdb.Database      // Prefixed table-view of the db to write index metadata into	索引存储的数据库
	backend  ChainIndexerBackend // Background processor generating the index data content	索引生成的后端。
	children []*ChainIndexer     // Child indexers to cascade chain updates to	子索引

	active    uint32          // Flag whether the event loop was started
	update    chan struct{}   // Notification channel that headers should be processed	接收到的headers
	quit      chan chan error // Quit channel to tear down running goroutines
	ctx       context.Context
	ctxCancel func()

	sectionSize uint64 // Number of blocks in a single chain segment to process	section的大小。 默认是4096个区块为一个section
	confirmsReq uint64 // Number of confirmations before processing a completed segment	处理完成的段之前的确认次数

	storedSections uint64 // Number of sections successfully indexed into the database	成功索引到数据库的部分数量
	knownSections  uint64 // Number of sections known to be complete (block wise)	已知完成的部分数量
	cascadedHead   uint64 // Block number of the last completed section cascaded to subindexers	级联到子索引的最后一个完成部分的块号

	checkpointSections uint64      // Number of sections covered by the checkpoint
	checkpointHead     common.Hash // Section head belonging to the checkpoint

	throttling time.Duration // Disk throttling to prevent a heavy upgrade from hogging resources	磁盘限制，以防止大量资源的大量升级

	log  log.Logger
	lock sync.Mutex
}

// NewChainIndexer creates a new chain indexer to do background processing on
// chain segments of a given size after certain number of confirmations passed.
// The throttling parameter might be used to prevent database thrashing.
func NewChainIndexer(chainDb ethdb.Database, indexDb ethdb.Database, backend ChainIndexerBackend, section, confirm uint64, throttling time.Duration, kind string) *ChainIndexer {
	c := &ChainIndexer{
		chainDb:     chainDb,
		indexDb:     indexDb,
		backend:     backend,
		update:      make(chan struct{}, 1),
		quit:        make(chan chan error),
		sectionSize: section,
		confirmsReq: confirm,
		throttling:  throttling,
		log:         log.New("type", kind),
	}
	// Initialize database dependent fields and start the updater
	c.loadValidSections()
	c.ctx, c.ctxCancel = context.WithCancel(context.Background())

	go c.updateLoop()

	return c
}

// AddCheckpoint adds a checkpoint. Sections are never processed and the chain
// is not expected to be available before this point. The indexer assumes that
// the backend has sufficient information available to process subsequent sections.
//
// Note: knownSections == 0 and storedSections == checkpointSections until
// syncing reaches the checkpoint
func (c *ChainIndexer) AddCheckpoint(section uint64, shead common.Hash) {
	c.lock.Lock()
	defer c.lock.Unlock()

	// Short circuit if the given checkpoint is below than local's.
	if c.checkpointSections >= section+1 || section < c.storedSections {
		return
	}
	c.checkpointSections = section + 1
	c.checkpointHead = shead

	c.setSectionHead(section, shead)
	c.setValidSections(section + 1)
}

// Start creates a goroutine to feed chain head events into the indexer for
// cascading background processing. Children do not need to be started, they
// are notified about new events by their parents.

// 子链不需要被启动。 以为他们的父节点会通知他们。
func (c *ChainIndexer) Start(chain ChainIndexerChain) {
	events := make(chan ChainHeadEvent, 10)
	sub := chain.SubscribeChainHeadEvent(events)

	go c.eventLoop(chain.CurrentHeader(), events, sub)
}

// Close tears down all goroutines belonging to the indexer and returns any error
// that might have occurred internally.
func (c *ChainIndexer) Close() error {
	var errs []error

	c.ctxCancel()

	// Tear down the primary update loop
	errc := make(chan error)
	c.quit <- errc
	if err := <-errc; err != nil {
		errs = append(errs, err)
	}
	// If needed, tear down the secondary event loop
	if atomic.LoadUint32(&c.active) != 0 {
		c.quit <- errc
		if err := <-errc; err != nil {
			errs = append(errs, err)
		}
	}
	// Close all children
	for _, child := range c.children {
		if err := child.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	// Return any failures
	switch {
	case len(errs) == 0:
		return nil

	case len(errs) == 1:
		return errs[0]

	default:
		return fmt.Errorf("%v", errs)
	}
}

// eventLoop is a secondary - optional - event loop of the indexer which is only
// started for the outermost indexer to push chain head events into a processing
// queue.

// eventLoop 循环只会在最外面的索引节点被调用。 所有的Child indexer不会被启动这个方法。

func (c *ChainIndexer) eventLoop(currentHeader *types.Header, events chan ChainHeadEvent, sub event.Subscription) {
	// Mark the chain indexer as active, requiring an additional teardown
	atomic.StoreUint32(&c.active, 1)

	defer sub.Unsubscribe()

	// Fire the initial new head event to start any outstanding processing
	// 设置我们的其实的区块高度，用来触发之前未完成的操作。
	c.newHead(currentHeader.Number.Uint64(), false)

	var (
		prevHeader = currentHeader
		prevHash   = currentHeader.Hash()
	)
	for {
		select {
		case errc := <-c.quit:
			// Chain indexer terminating, report no failure and abort
			errc <- nil
			return

		case ev, ok := <-events:
			// Received a new event, ensure it's not nil (closing) and update
			if !ok {
				errc := <-c.quit
				errc <- nil
				return
			}
			header := ev.Block.Header()
			if header.ParentHash != prevHash {//如果出现了分叉，那么我们首先
				// 找到公共祖先， 从公共祖先之后的索引需要重建。
				// Reorg to the common ancestor if needed (might not exist in light sync mode, skip reorg then)
				// TODO(karalabe, zsfelfoldi): This seems a bit brittle, can we detect this case explicitly?

				if rawdb.ReadCanonicalHash(c.chainDb, prevHeader.Number.Uint64()) != prevHash {
					if h := rawdb.FindCommonAncestor(c.chainDb, prevHeader, header); h != nil {
						c.newHead(h.Number.Uint64(), true)
					}
				}
			}
			// 设置新的head
			c.newHead(header.Number.Uint64(), false)

			prevHeader, prevHash = header, header.Hash()
		}
	}
}

// newHead notifies the indexer about new chain heads and/or reorgs.
// newHead方法,通知indexer新的区块链头，或者是需要重建索引，newHead方法会触发
func (c *ChainIndexer) newHead(head uint64, reorg bool) {
	c.lock.Lock()
	defer c.lock.Unlock()

	// If a reorg happened, invalidate all sections until that point
	if reorg { // 需要重建索引 从head开始的所有section都需要重建。
		// Revert the known section number to the reorg point
		known := (head + 1) / c.sectionSize
		stored := known
		if known < c.checkpointSections {
			known = 0
		}
		if stored < c.checkpointSections {
			stored = c.checkpointSections
		}
		if known < c.knownSections {
			c.knownSections = known
		}
		// Revert the stored sections from the database to the reorg point
		// 将存储的部分从数据库恢复到索引重建点
		if stored < c.storedSections {
			c.setValidSections(stored)
		}
		// Update the new head number to the finalized section end and notify children
		// 生成新的head 并通知所有的子索引
		head = known * c.sectionSize

		if head < c.cascadedHead {
			c.cascadedHead = head
			for _, child := range c.children {
				child.newHead(c.cascadedHead, true)
			}
		}
		return
	}
	// No reorg, calculate the number of newly known sections and update if high enough
	var sections uint64
	if head >= c.confirmsReq {
		sections = (head + 1 - c.confirmsReq) / c.sectionSize
		if sections < c.checkpointSections {
			sections = 0
		}
		if sections > c.knownSections {
			if c.knownSections < c.checkpointSections {
				// syncing reached the checkpoint, verify section head
				syncedHead := rawdb.ReadCanonicalHash(c.chainDb, c.checkpointSections*c.sectionSize-1)
				if syncedHead != c.checkpointHead {
					c.log.Error("Synced chain does not match checkpoint", "number", c.checkpointSections*c.sectionSize-1, "expected", c.checkpointHead, "synced", syncedHead)
					return
				}
			}
			c.knownSections = sections

			select {
			case c.update <- struct{}{}:
			default:
			}
		}
	}
}

// updateLoop is the main event loop of the indexer which pushes chain segments
// down into the processing backend.
// updateLoop,是主要的事件循环，用于调用backend来处理区块链section，这个需要注意的是，所有的主索引节点和所有的 child indexer 都会启动这个goroutine 方法
func (c *ChainIndexer) updateLoop() {
	var (
		updating bool
		updated  time.Time
	)

	for {
		select {
		case errc := <-c.quit:
			// Chain indexer terminating, report no failure and abort
			errc <- nil
			return

		case <-c.update: //当需要使用backend处理的时候，其他goroutine会往这个channel上面发送消息
			// Section headers completed (or rolled back), update the index
			c.lock.Lock()
			if c.knownSections > c.storedSections { // 如果当前以知的Section 大于已经存储的Section
				// Periodically print an upgrade log message to the user
				// 每隔8秒打印一次日志信息。
				if time.Since(updated) > 8*time.Second {
					if c.knownSections > c.storedSections+1 {
						updating = true
						c.log.Info("Upgrading chain index", "percentage", c.storedSections*100/c.knownSections)
					}
					updated = time.Now()
				}
				// Cache the current section count and head to allow unlocking the mutex
				c.verifyLastHead()
				section := c.storedSections
				var oldHead common.Hash
				if section > 0 { // section - 1 代表section的下标是从0开始的。
					// sectionHead用来获取section的最后一个区块的hash值。
					oldHead = c.SectionHead(section - 1)
				}
				// Process the newly defined section in the background
				c.lock.Unlock()
				// 处理 返回新的section的最后一个区块的hash值
				newHead, err := c.processSection(section, oldHead)
				if err != nil {
					select {
					case <-c.ctx.Done():
						<-c.quit <- nil
						return
					default:
					}
					c.log.Error("Section processing failed", "error", err)
				}
				c.lock.Lock()

				// If processing succeeded and no reorgs occurred, mark the section completed
				if err == nil && (section == 0 || oldHead == c.SectionHead(section-1)) {
					c.setSectionHead(section, newHead)  // 更新数据库的状态
					c.setValidSections(section + 1)	// 更新数据库状态
					if c.storedSections == c.knownSections && updating {
						updating = false
						c.log.Info("Finished upgrading chain index")
					}
					// cascadedHead 是更新后的section的最后一个区块的高度
					c.cascadedHead = c.storedSections*c.sectionSize - 1
					for _, child := range c.children {
						c.log.Trace("Cascading chain index update", "head", c.cascadedHead)
						child.newHead(c.cascadedHead, false)
					}
				} else {
					//如果处理失败，那么在有新的通知之前不会重试。
					// If processing failed, don't retry until further notification
					c.log.Debug("Chain index processing failed", "section", section, "err", err)
					c.verifyLastHead()
					c.knownSections = c.storedSections
				}
			}
			// If there are still further sections to process, reschedule
			// 如果还有section等待处理，那么等待throttling时间再处理。避免磁盘过载。
			if c.knownSections > c.storedSections {
				time.AfterFunc(c.throttling, func() {
					select {
					case c.update <- struct{}{}:
					default:
					}
				})
			}
			c.lock.Unlock()
		}
	}
}

// processSection processes an entire section by calling backend functions while
// ensuring the continuity of the passed headers. Since the chain mutex is not
// held while processing, the continuity can be broken by a long reorg, in which
// case the function returns with an error.

// processSection通过调用后端函数来处理整个部分，同时确保传递的头文件的连续性。 由于链接互斥锁在处理过程中没有保持，连续性可能会被重新打断，在这种情况下，函数返回一个错误。
func (c *ChainIndexer) processSection(section uint64, lastHead common.Hash) (common.Hash, error) {
	c.log.Trace("Processing new chain section", "section", section)

	// Reset and partial processing
	if err := c.backend.Reset(c.ctx, section, lastHead); err != nil {
		c.setValidSections(0)
		return common.Hash{}, err
	}

	for number := section * c.sectionSize; number < (section+1)*c.sectionSize; number++ {
		hash := rawdb.ReadCanonicalHash(c.chainDb, number)
		if hash == (common.Hash{}) {
			return common.Hash{}, fmt.Errorf("canonical block #%d unknown", number)
		}
		header := rawdb.ReadHeader(c.chainDb, hash, number)
		if header == nil {
			return common.Hash{}, fmt.Errorf("block #%d [%x..] not found", number, hash[:4])
		} else if header.ParentHash != lastHead {
			return common.Hash{}, fmt.Errorf("chain reorged during section processing")
		}
		if err := c.backend.Process(c.ctx, header); err != nil {
			return common.Hash{}, err
		}
		lastHead = header.Hash()
	}
	if err := c.backend.Commit(); err != nil {
		return common.Hash{}, err
	}
	return lastHead, nil
}

// verifyLastHead compares last stored section head with the corresponding block hash in the
// actual canonical chain and rolls back reorged sections if necessary to ensure that stored
// sections are all valid
func (c *ChainIndexer) verifyLastHead() {
	for c.storedSections > 0 && c.storedSections > c.checkpointSections {
		if c.SectionHead(c.storedSections-1) == rawdb.ReadCanonicalHash(c.chainDb, c.storedSections*c.sectionSize-1) {
			return
		}
		c.setValidSections(c.storedSections - 1)
	}
}

// Sections returns the number of processed sections maintained by the indexer
// and also the information about the last header indexed for potential canonical
// verifications.
func (c *ChainIndexer) Sections() (uint64, uint64, common.Hash) {
	c.lock.Lock()
	defer c.lock.Unlock()

	c.verifyLastHead()
	return c.storedSections, c.storedSections*c.sectionSize - 1, c.SectionHead(c.storedSections - 1)
}

// AddChildIndexer adds a child ChainIndexer that can use the output of this one
func (c *ChainIndexer) AddChildIndexer(indexer *ChainIndexer) {
	if indexer == c {
		panic("can't add indexer as a child of itself")
	}
	c.lock.Lock()
	defer c.lock.Unlock()

	c.children = append(c.children, indexer)

	// Cascade any pending updates to new children too
	sections := c.storedSections
	if c.knownSections < sections {
		// if a section is "stored" but not "known" then it is a checkpoint without
		// available chain data so we should not cascade it yet
		sections = c.knownSections
	}
	if sections > 0 {
		indexer.newHead(sections*c.sectionSize-1, false)
	}
}

// Prune deletes all chain data older than given threshold.
func (c *ChainIndexer) Prune(threshold uint64) error {
	return c.backend.Prune(threshold)
}

// loadValidSections reads the number of valid sections from the index database
// and caches is into the local state.
// loadValidSections,用来从数据库里面加载我们之前的处理信息， storedSections表示我们已经处理到哪里了。
func (c *ChainIndexer) loadValidSections() {
	data, _ := c.indexDb.Get([]byte("count"))
	if len(data) == 8 {
		c.storedSections = binary.BigEndian.Uint64(data)
	}
}

// setValidSections writes the number of valid sections to the index database
// 写入当前已经存储的sections的数量。 如果传入的值小于已经存储的数量，那么从数据库里面删除对应的section
func (c *ChainIndexer) setValidSections(sections uint64) {
	// Set the current number of valid sections in the database
	var data [8]byte
	binary.BigEndian.PutUint64(data[:], sections)
	c.indexDb.Put([]byte("count"), data[:])

	// Remove any reorged sections, caching the valids in the mean time
	for c.storedSections > sections {
		c.storedSections--
		c.removeSectionHead(c.storedSections)
	}
	c.storedSections = sections // needed if new > old
}

// SectionHead retrieves the last block hash of a processed section from the
// index database.
func (c *ChainIndexer) SectionHead(section uint64) common.Hash {
	var data [8]byte
	binary.BigEndian.PutUint64(data[:], section)

	hash, _ := c.indexDb.Get(append([]byte("shead"), data[:]...))
	if len(hash) == len(common.Hash{}) {
		return common.BytesToHash(hash)
	}
	return common.Hash{}
}

// setSectionHead writes the last block hash of a processed section to the index
// database.
func (c *ChainIndexer) setSectionHead(section uint64, hash common.Hash) {
	var data [8]byte
	binary.BigEndian.PutUint64(data[:], section)

	c.indexDb.Put(append([]byte("shead"), data[:]...), hash.Bytes())
}

// removeSectionHead removes the reference to a processed section from the index
// database.
func (c *ChainIndexer) removeSectionHead(section uint64) {
	var data [8]byte
	binary.BigEndian.PutUint64(data[:], section)

	c.indexDb.Delete(append([]byte("shead"), data[:]...))
}
