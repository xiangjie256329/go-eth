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

package bloombits

import (
	"bytes"
	"context"
	"errors"
	"math"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ethereum/go-ethereum/common/bitutil"
	"github.com/ethereum/go-ethereum/crypto"
)

// bloomIndexes represents the bit indexes inside the bloom filter that belong
// to some key.
type bloomIndexes [3]uint

// calcBloomIndexes returns the bloom filter bit indexes belonging to the given key.
func calcBloomIndexes(b []byte) bloomIndexes {
	b = crypto.Keccak256(b)

	var idxs bloomIndexes
	for i := 0; i < len(idxs); i++ {
		idxs[i] = (uint(b[2*i])<<8)&2047 + uint(b[2*i+1])
	}
	return idxs
}

// partialMatches with a non-nil vector represents a section in which some sub-
// matchers have already found potential matches. Subsequent sub-matchers will
// binary AND their matches with this vector. If vector is nil, it represents a
// section to be processed by the first sub-matcher.
// partialMatches代表了部分匹配的结果。 比入有三个需要过滤的条件 addr1, addr2, addr3 ，需要找到同时匹配这三个条件的数据。 那么我们启动包含了匹配这三个条件的流水线。
// 第一个匹配的结果会送给第二个，第二个把第一个的结果和自己的结果执行bit与操作，然后作为匹配的结果送给第三个处理。
type partialMatches struct {
	section uint64
	bitset  []byte
}

// Retrieval represents a request for retrieval task assignments for a given
// bit with the given number of fetch elements, or a response for such a request.
// It can also have the actual results set to be used as a delivery data struct.
//
// The contest and error fields are used by the light client to terminate matching
// early if an error is encountered on some path of the pipeline.
// Retrieval 代表了 一次区块布隆过滤器索引的检索工作， 这个对象被发送给 eth/bloombits.go 里面的 startBloomHandlers来处理，
// 这个方法从数据库来加载布隆过滤器索引然后放在Bitsets里面返回。
type Retrieval struct {
	Bit      uint
	Sections []uint64
	Bitsets  [][]byte

	Context context.Context
	Error   error
}

// Matcher is a pipelined system of schedulers and logic matchers which perform
// binary AND/OR operations on the bit-streams, creating a stream of potential
// blocks to inspect for data content.
type Matcher struct {
	sectionSize uint64 // Size of the data batches to filter on

	filters    [][]bloomIndexes    // Filter the system is matching for
	schedulers map[uint]*scheduler // Retrieval schedulers for loading bloom bits

	retrievers chan chan uint       // Retriever processes waiting for bit allocations	用来传递 检索任务的通道
	counters   chan chan uint       // Retriever processes waiting for task count reports	用来返回当前所有的任务数量
	retrievals chan chan *Retrieval // Retriever processes waiting for task allocations	用来传递 检索任务的分配
	deliveries chan *Retrieval      // Retriever processes waiting for task response deliveries	检索完成的结果传递到这个通道

	running uint32 // Atomic flag whether a session is live or not
}

// NewMatcher creates a new pipeline for retrieving bloom bit streams and doing
// address and topic filtering on them. Setting a filter component to `nil` is
// allowed and will result in that filter rule being skipped (OR 0x11...1).
func NewMatcher(sectionSize uint64, filters [][][]byte) *Matcher {
	// Create the matcher instance
	m := &Matcher{
		sectionSize: sectionSize,
		schedulers:  make(map[uint]*scheduler),
		retrievers:  make(chan chan uint),
		counters:    make(chan chan uint),
		retrievals:  make(chan chan *Retrieval),
		deliveries:  make(chan *Retrieval),
	}
	// Calculate the bloom bit indexes for the groups we're interested in
	m.filters = nil

	for _, filter := range filters {
		// Gather the bit indexes of the filter rule, special casing the nil filter
		if len(filter) == 0 {
			continue
		}
		bloomBits := make([]bloomIndexes, len(filter))
		for i, clause := range filter {
			if clause == nil {
				bloomBits = nil
				break
			}
			// clause 对应了输入的第三维度的数据，可能是一个address或者是一个topic
			// calcBloomIndexes计算了这个数据对应的(0-2048)的布隆过滤器中的三个下标， 就是说如果在布隆过滤器中对应的三位都为1，那么clause这个数据就有可能在这里。
			bloomBits[i] = calcBloomIndexes(clause)
		}
		// Accumulate the filter rules if no nil rule was within
		// 在计算中 如果bloomBits中只要其中的一条能够找到。那么就认为整个成立。
		if bloomBits != nil {
			// 不同的bloomBits 需要同时成立，整个结果才能成立。
			m.filters = append(m.filters, bloomBits)
		}
	}
	// For every bit, create a scheduler to load/download the bit vectors
	for _, bloomIndexLists := range m.filters {
		for _, bloomIndexList := range bloomIndexLists {
			for _, bloomIndex := range bloomIndexList {
				// 对于所有可能出现的下标。 我们都生成一个scheduler来进行对应位置的
				// 布隆过滤数据的检索。
				m.addScheduler(bloomIndex)
			}
		}
	}
	return m
}

// addScheduler adds a bit stream retrieval scheduler for the given bit index if
// it has not existed before. If the bit is already selected for filtering, the
// existing scheduler can be used.
func (m *Matcher) addScheduler(idx uint) {
	if _, ok := m.schedulers[idx]; ok {
		return
	}
	m.schedulers[idx] = newScheduler(idx)
}

// Start starts the matching process and returns a stream of bloom matches in
// a given range of blocks. If there are no more matches in the range, the result
// channel is closed.
func (m *Matcher) Start(ctx context.Context, begin, end uint64, results chan uint64) (*MatcherSession, error) {
	// Make sure we're not creating concurrent sessions
	if atomic.SwapUint32(&m.running, 1) == 1 {
		return nil, errors.New("matcher already running")
	}
	defer atomic.StoreUint32(&m.running, 0)

	// Initiate a new matching round
	// 启动了一个session，作为返回值，管理查找的生命周期。
	session := &MatcherSession{
		matcher: m,
		quit:    make(chan struct{}),
		ctx:     ctx,
	}
	for _, scheduler := range m.schedulers {
		scheduler.reset()
	}
	// 这个运行会建立起流程，返回了一个partialMatches类型的管道表示查询的部分结果。
	sink := m.run(begin, end, cap(results), session)

	// Read the output from the result sink and deliver to the user
	session.pend.Add(1)
	go func() {
		defer session.pend.Done()
		defer close(results)

		for {
			select {
			case <-session.quit:
				return

			case res, ok := <-sink:
				// New match result found
				// 找到返回结果 因为返回值是 section和 section中哪些区块可能有值的bitmap
				// 所以需要遍历这个bitmap，找到那些被置位的区块，把区块号返回回去。
				if !ok {
					return
				}
				// Calculate the first and last blocks of the section
				sectionStart := res.section * m.sectionSize

				first := sectionStart
				if begin > first {
					first = begin
				}
				last := sectionStart + m.sectionSize - 1
				if end < last {
					last = end
				}
				// Iterate over all the blocks in the section and return the matching ones
				for i := first; i <= last; i++ {
					// Skip the entire byte if no matches are found inside (and we're processing an entire byte!)
					next := res.bitset[(i-sectionStart)/8]
					if next == 0 {
						if i%8 == 0 {
							i += 7
						}
						continue
					}
					// Some bit it set, do the actual submatching
					if bit := 7 - i%8; next&(1<<bit) != 0 {
						select {
						case <-session.quit:
							return
						case results <- i:
						}
					}
				}
			}
		}
	}()
	return session, nil
}

// run creates a daisy-chain of sub-matchers, one for the address set and one
// for each topic set, each sub-matcher receiving a section only if the previous
// ones have all found a potential match in one of the blocks of the section,
// then binary AND-ing its own matches and forwarding the result to the next one.
//	创建一个子匹配器的流水线，一个用于地址集，一个用于每个主题集，每个子匹配器只有在先前的所有子块都在该部分的一个块中找到可能的匹配时才接收一个部分，然后把接收到的和自己的匹配，并将结果转交给下一个。
// The method starts feeding the section indexes into the first sub-matcher on a
// new goroutine and returns a sink channel receiving the results.
// 该方法开始section indexer送到第一个子匹配器，并返回接收结果的接收器通道。
func (m *Matcher) run(begin, end uint64, buffer int, session *MatcherSession) chan *partialMatches {
	// Create the source channel and feed section indexes into
	source := make(chan *partialMatches, buffer)

	session.pend.Add(1)
	go func() {
		defer session.pend.Done()
		defer close(source)

		for i := begin / m.sectionSize; i <= end/m.sectionSize; i++ {
			// 这个for循环 构造了subMatch的第一个输入源，剩下的subMatch把上一个的结果作为自己的源
			// 这个源的bitset字段都是0xff，代表完全的匹配，它将和我们这一步的匹配进行与操作，得到这一步匹配的结果。
			select {
			case <-session.quit:
				return
			case source <- &partialMatches{i, bytes.Repeat([]byte{0xff}, int(m.sectionSize/8))}:
			}
		}
	}()
	// Assemble the daisy-chained filtering pipeline
	next := source
	dist := make(chan *request, buffer)

	for _, bloom := range m.filters {	//构建流水线， 前一个的输出作为下一个subMatch的输入。
		next = m.subMatch(next, dist, bloom, session)
	}
	// Start the request distribution
	session.pend.Add(1)
	// 启动distributor线程。
	go m.distributor(dist, session)

	return next
}

// subMatch creates a sub-matcher that filters for a set of addresses or topics, binary OR-s those matches, then
// binary AND-s the result to the daisy-chain input (source) and forwards it to the daisy-chain output.
// The matches of each address/topic are calculated by fetching the given sections of the three bloom bit indexes belonging to
// that address/topic, and binary AND-ing those vectors together.
// subMatch创建一个子匹配器，用于过滤一组地址或主题，对这些主题进行bit位或操作，然后将上一个结果与当前过滤结果进行位与操作，如果结果不全位空，就把结果传递给下一个子匹配器。
// 每个地址/题目的匹配是通过获取属于该地址/题目的三个布隆过滤器位索引的给定部分以及将这些向量二进制AND并在一起来计算的。
// subMatch是最重要的一个函数， 把filters [][][3]的 第一维度的与，第二维度的或， 第三维度的与操作 结合在一起。
func (m *Matcher) subMatch(source chan *partialMatches, dist chan *request, bloom []bloomIndexes, session *MatcherSession) chan *partialMatches {
	// Start the concurrent schedulers for each bit required by the bloom filter
	// 传入的bloom []bloomIndexes参数是filters的第二,第三维度  [][3]
	sectionSources := make([][3]chan uint64, len(bloom))
	sectionSinks := make([][3]chan []byte, len(bloom))
	for i, bits := range bloom {	// i代表了第二维度的数量
		for j, bit := range bits {	//j 代表了布隆过滤器的下标 肯定只有三个 取值(0-2048)
			sectionSources[i][j] = make(chan uint64, cap(source))	// 创建scheduler的输入channel
			sectionSinks[i][j] = make(chan []byte, cap(source))	// 创建 scheduler的输出channel

			// 对这个bit发起调度请求， 通过sectionSources[i][j]传递需要查询的section
			// 通过sectionSinks[i][j]来接收结果
			// dist 是scheduler传递请求的通道。 这个在scheduler的介绍里面有。
			m.schedulers[bit].run(sectionSources[i][j], dist, sectionSinks[i][j], session.quit, &session.pend)
		}
	}

	process := make(chan *partialMatches, cap(source)) // entries from source are forwarded here after fetches have been initiated
	results := make(chan *partialMatches, cap(source))	// 返回值channel

	session.pend.Add(2)
	go func() {
		// Tear down the goroutine and terminate all source channels
		defer session.pend.Done()
		defer close(process)

		defer func() {
			for _, bloomSources := range sectionSources {
				for _, bitSource := range bloomSources {
					close(bitSource)
				}
			}
		}()
		// Read sections from the source channel and multiplex into all bit-schedulers
		// 从source channel读取sections 并把这些数据通过sectionSources传递给scheduler
		for {
			select {
			case <-session.quit:
				return

			case subres, ok := <-source:
				// New subresult from previous link
				if !ok {
					return
				}
				// Multiplex the section index to all bit-schedulers
				for _, bloomSources := range sectionSources {
					for _, bitSource := range bloomSources {
						// 传递给上面的所有的scheduler的输入通道。 申请对这些
						// section 的指定bit进行查找。 结果会发送给sectionSinks[i][j]
						select {
						case <-session.quit:
							return
						case bitSource <- subres.section:
						}
					}
				}
				// Notify the processor that this section will become available
				select {
				case <-session.quit:
					return
				case process <- subres:	//等到所有的请求都递交给scheduler 给process发送消息。
				}
			}
		}
	}()

	go func() {
		// Tear down the goroutine and terminate the final sink channel
		defer session.pend.Done()
		defer close(results)

		// Read the source notifications and collect the delivered results
		for {
			select {
			case <-session.quit:
				return

			case subres, ok := <-process:
				// Notified of a section being retrieved
				// 这里有个问题。 有没有可能乱序。 因为通道都是有缓存的。 可能查询得快慢导致
				// 查看了scheduler的实现， scheduler是保证顺序的。怎么进来，就会怎么出去。
				if !ok {
					return
				}
				// Gather all the sub-results and merge them together
				var orVector []byte
				for _, bloomSinks := range sectionSinks {
					var andVector []byte
					for _, bitSink := range bloomSinks { // 这里可以接收到三个值 每个代表了对应下标的 布隆过滤器的值,对这三个值进行与操作，就可以得到那些区块可能存在对应的值。
						var data []byte
						select {
						case <-session.quit:
							return
						case data = <-bitSink:
						}
						if andVector == nil {
							andVector = make([]byte, int(m.sectionSize/8))
							copy(andVector, data)
						} else {
							bitutil.ANDBytes(andVector, andVector, data)
						}
					}
					if orVector == nil { //对第一维度的数据执行 Or操作。
						orVector = andVector
					} else {
						bitutil.ORBytes(orVector, orVector, andVector)
					}
				}

				if orVector == nil {	//可能通道被关闭了。 没有查询到任何值
					orVector = make([]byte, int(m.sectionSize/8))
				}
				if subres.bitset != nil {
					// 和输入的上一次的结果进行与操作。 记得最开始这个值被初始化为全1
					bitutil.ANDBytes(orVector, orVector, subres.bitset)
				}
				if bitutil.TestBytes(orVector) {	// 如果不全为0 那么添加到结果。可能会给下一个匹配。或者是返回。
					select {
					case <-session.quit:
						return
					case results <- &partialMatches{subres.section, orVector}:
					}
				}
			}
		}
	}()
	return results
}

// distributor receives requests from the schedulers and queues them into a set
// of pending requests, which are assigned to retrievers wanting to fulfil them.
func (m *Matcher) distributor(dist chan *request, session *MatcherSession) {
	defer session.pend.Done()

	var (
		requests   = make(map[uint][]uint64) // Per-bit list of section requests, ordered by section number
		unallocs   = make(map[uint]struct{}) // Bits with pending requests but not allocated to any retriever
		retrievers chan chan uint            // Waiting retrievers (toggled to nil if unallocs is empty)
		allocs     int                       // Number of active allocations to handle graceful shutdown requests
		shutdown   = session.quit            // Shutdown request channel, will gracefully wait for pending requests
	)

	// assign is a helper method fo try to assign a pending bit an actively
	// listening servicer, or schedule it up for later when one arrives.
	assign := func(bit uint) {
		select {
		case fetcher := <-m.retrievers:
			allocs++
			fetcher <- bit
		default:
			// No retrievers active, start listening for new ones
			retrievers = m.retrievers
			unallocs[bit] = struct{}{}
		}
	}

	for {
		select {
		case <-shutdown:
			// Shutdown requested. No more retrievers can be allocated,
			// but we still need to wait until all pending requests have returned.
			shutdown = nil
			if allocs == 0 {
				return
			}

		case req := <-dist: // scheduler发送来的请求 添加到指定bit位置的queue里面
			// New retrieval request arrived to be distributed to some fetcher process
			queue := requests[req.bit]
			index := sort.Search(len(queue), func(i int) bool { return queue[i] >= req.section })
			requests[req.bit] = append(queue[:index], append([]uint64{req.section}, queue[index:]...)...)

			// If it's a new bit and we have waiting fetchers, allocate to them
			// 如果这个bit是一个新的。 还没有被指派，那么我们把他指派给等待的fetchers
			if len(queue) == 0 {
				assign(req.bit)
			}

		case fetcher := <-retrievers:
			// New retriever arrived, find the lowest section-ed bit to assign
			// 如果新的retrievers进来了， 那么我们查看是否有任务没有指派
			bit, best := uint(0), uint64(math.MaxUint64)
			for idx := range unallocs {
				if requests[idx][0] < best {
					bit, best = idx, requests[idx][0]
				}
			}
			// Stop tracking this bit (and alloc notifications if no more work is available)
			delete(unallocs, bit)
			if len(unallocs) == 0 { //如果所有任务都被指派。那么停止关注retrievers
				retrievers = nil
			}
			allocs++
			fetcher <- bit

		case fetcher := <-m.counters:
			// New task count request arrives, return number of items
			// 来了新的请求，访问request的指定bit的数量。
			fetcher <- uint(len(requests[<-fetcher]))

		case fetcher := <-m.retrievals:
			// New fetcher waiting for tasks to retrieve, assign
			// 有人来领取任务。
			task := <-fetcher
			if want := len(task.Sections); want >= len(requests[task.Bit]) {
				task.Sections = requests[task.Bit]
				delete(requests, task.Bit)
			} else {
				task.Sections = append(task.Sections[:0], requests[task.Bit][:want]...)
				requests[task.Bit] = append(requests[task.Bit][:0], requests[task.Bit][want:]...)
			}
			fetcher <- task

			// If anything was left unallocated, try to assign to someone else
			// 如果还有任务没有分派完。 尝试分配给其他人。
			if len(requests[task.Bit]) > 0 {
				assign(task.Bit)
			}

		case result := <-m.deliveries:
			// New retrieval task response from fetcher, split out missing sections and
			// deliver complete ones
			// 收到了任务的结果。
			var (
				sections = make([]uint64, 0, len(result.Sections))
				bitsets  = make([][]byte, 0, len(result.Bitsets))
				missing  = make([]uint64, 0, len(result.Sections))
			)
			for i, bitset := range result.Bitsets {
				if len(bitset) == 0 {	//如果任务结果有缺失，记录下来
					missing = append(missing, result.Sections[i])
					continue
				}
				sections = append(sections, result.Sections[i])
				bitsets = append(bitsets, bitset)
			}
			// 投递结果
			m.schedulers[result.Bit].deliver(sections, bitsets)
			allocs--

			// Reschedule missing sections and allocate bit if newly available
			if len(missing) > 0 {	//如果有缺失， 那么重新生成新的任务。
				queue := requests[result.Bit]
				for _, section := range missing {
					index := sort.Search(len(queue), func(i int) bool { return queue[i] >= section })
					queue = append(queue[:index], append([]uint64{section}, queue[index:]...)...)
				}
				requests[result.Bit] = queue

				if len(queue) == len(missing) {
					assign(result.Bit)
				}
			}

			// End the session when all pending deliveries have arrived.
			if shutdown == nil && allocs == 0 {
				return
			}
		}
	}
}

// MatcherSession is returned by a started matcher to be used as a terminator
// for the actively running matching operation.
type MatcherSession struct {
	matcher *Matcher

	closer sync.Once     // Sync object to ensure we only ever close once
	quit   chan struct{} // Quit channel to request pipeline termination

	ctx     context.Context // Context used by the light client to abort filtering
	err     error           // Global error to track retrieval failures deep in the chain
	errLock sync.Mutex

	pend sync.WaitGroup
}

// Close stops the matching process and waits for all subprocesses to terminate
// before returning. The timeout may be used for graceful shutdown, allowing the
// currently running retrievals to complete before this time.
func (s *MatcherSession) Close() {
	s.closer.Do(func() {
		// Signal termination and wait for all goroutines to tear down
		close(s.quit)
		s.pend.Wait()
	})
}

// Error returns any failure encountered during the matching session.
func (s *MatcherSession) Error() error {
	s.errLock.Lock()
	defer s.errLock.Unlock()

	return s.err
}

// allocateRetrieval assigns a bloom bit index to a client process that can either
// immediately request and fetch the section contents assigned to this bit or wait
// a little while for more sections to be requested.
func (s *MatcherSession) allocateRetrieval() (uint, bool) {
	fetcher := make(chan uint)

	select {
	case <-s.quit:
		return 0, false
	case s.matcher.retrievers <- fetcher:
		bit, ok := <-fetcher
		return bit, ok
	}
}

// pendingSections returns the number of pending section retrievals belonging to
// the given bloom bit index.
func (s *MatcherSession) pendingSections(bit uint) int {
	fetcher := make(chan uint)

	select {
	case <-s.quit:
		return 0
	case s.matcher.counters <- fetcher:
		fetcher <- bit
		return int(<-fetcher)
	}
}

// allocateSections assigns all or part of an already allocated bit-task queue
// to the requesting process.
func (s *MatcherSession) allocateSections(bit uint, count int) []uint64 {
	fetcher := make(chan *Retrieval)

	select {
	case <-s.quit:
		return nil
	case s.matcher.retrievals <- fetcher:
		task := &Retrieval{
			Bit:      bit,
			Sections: make([]uint64, count),
		}
		fetcher <- task
		return (<-fetcher).Sections
	}
}

// deliverSections delivers a batch of section bit-vectors for a specific bloom
// bit index to be injected into the processing pipeline.
func (s *MatcherSession) deliverSections(bit uint, sections []uint64, bitsets [][]byte) {
	s.matcher.deliveries <- &Retrieval{Bit: bit, Sections: sections, Bitsets: bitsets}
}

// Multiplex polls the matcher session for retrieval tasks and multiplexes it into
// the requested retrieval queue to be serviced together with other sessions.
//
// This method will block for the lifetime of the session. Even after termination
// of the session, any request in-flight need to be responded to! Empty responses
// are fine though in that case.
// 任务的执行Multiplex,Multiplex函数不断的领取任务，把任务投递给bloomRequest队列。从队列获取结果。然后投递给distributor。 完成了整个过程。
func (s *MatcherSession) Multiplex(batch int, wait time.Duration, mux chan chan *Retrieval) {
	for {
		// Allocate a new bloom bit index to retrieve data for, stopping when done
		bit, ok := s.allocateRetrieval()
		if !ok {
			return
		}
		// Bit allocated, throttle a bit if we're below our batch limit
		if s.pendingSections(bit) < batch {
			select {
			case <-s.quit:
				// Session terminating, we can't meaningfully service, abort
				s.allocateSections(bit, 0)
				s.deliverSections(bit, []uint64{}, [][]byte{})
				return

			case <-time.After(wait):
				// Throttling up, fetch whatever's available
			}
		}
		// Allocate as much as we can handle and request servicing
		sections := s.allocateSections(bit, batch)
		request := make(chan *Retrieval)

		select {
		case <-s.quit:
			// Session terminating, we can't meaningfully service, abort
			s.deliverSections(bit, sections, make([][]byte, len(sections)))
			return

		case mux <- request:
			// Retrieval accepted, something must arrive before we're aborting
			request <- &Retrieval{Bit: bit, Sections: sections, Context: s.ctx}

			result := <-request
			if result.Error != nil {
				s.errLock.Lock()
				s.err = result.Error
				s.errLock.Unlock()
				s.Close()
			}
			s.deliverSections(result.Bit, result.Sections, result.Bitsets)
		}
	}
}
