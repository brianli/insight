         ![[Y2026/Q3/0804_KVCache/assets/Image 5.webp|Image]]

![[Y2026/Q3/0804_KVCache/assets/Image 6.webp|Image]]

https://mp.weixin.qq.com/s/_DzOw9UJZfbGWQPHMFlRrg
#AIInfra  该公众号下有着很多AI Infra 相关的文章

作者介绍：一名正在持续在做大模型统一编排框架(同时支持Dynamo/llm-d/AIBrix/自研)与统一多推理引擎(vllm/sglang/...及多种KVCache后端方案集群方案的工程实践者，当前主要关注并推进统一推理编排层、多后端 serving 抽象、P/D 分离与 runtime 数据面验证。而笔者在2015-2019年在云存储领域从事基于纠删码RS视频图片数据存储，主要负责视频元数据(索引)及磁盘调度部分，单区域节点接入10万数量级视频设备实时存储，存储容量10PB左右，我们当时技术方案是对实时视频采用时间第一级分片切割(如30s)，第二级基于RS编码切割(即30s视频数据段按M+N RS分片)在传统存储还是副本存储方式具有非常高的存储成本优势，但是由于基于视频内的用户场景不应该感知对象，切片，在元数据服务上需要对磁盘存储服务上报的分段列表进行对外进行时间轴合并，对内需要保持严格的索引序列，另外由于视频海量实时存储随着 运行周期拉长磁盘损坏是常态，磁盘调度需要对磁盘实时健康度监控，在冗余度不够时需要历史数据迁移，对实时存储需要动态换盘甚至换组，在从流媒体服务转发的数据需要有抗写入异常能力，所以也存在内存，ssd, 磁盘服务的多级卸载过程，所以在初次接触分布式KVCache代表产品Mooncake时，非常自然感觉它们是惊人相似，甚至部分程序中的变量，流程高度重叠，本文不是去强行对比，而只是作者的一次自然知识复盘或者系统结构化过程，主旨不在深潜分布式kvcache 或者mooncake的具体细节，而只是在这个门口体验横看成岭侧成峰，远近高低各不同的工程思考。

Mooncake Store 处在三套参照系之间：经典分布式对象存储提供对象粒度、放置、数据面和后台治理经验；经典 KV 存储提供 key 定位、分片、副本和失效语义；KVCache 分布式存储把 value 改写成 GPU 计算中间态，并把目标函数推向 TTFT、命中率、重算成本和拓扑搬运。所以这里需要警惕读者看到相似性的同时需要其间本质差异。

01

TL;DR

- 对象价值变化： 经典对象存储保存长期业务数据，而经典 KV 存储保存业务值或状态，KVCache 则保存 GPU 计算过的中间态。一次 miss 不是普通回源，而可能触发 prefill 重算、TTFT 上升和 GPU 利用率下降。
    

- 存储骨架没变： 对象粒度、索引粒度、放置语义、数据面、故障语义仍然是主线。Mooncake 只是把对象从文件/视频切片换成 tensor slice，把介质从 DRAM/SSD/HDD 扩展到 GPU VRAM/HBM。
    

- 控制面退出热链路： mooncake_master管理 client、segment、object、replica、TTL、eviction 和 task；KVCache 搬运由 Transfer Engine 在 store client 之间完成。
    

- 经典分布式对象存储系统对理解分步式KVCache有参照价值： Ceph 说明 placement 是可计算能力；Swift 说明后台治理是对象系统主负载；FastDFS 提供 tracker/storage 的朴素边界；TFS 说明业务对象粒度和物理分配粒度必须分离；云对象存储说明成本函数会塑造产品形态。
    

- 经典 KV 存储提供另一半参照： Redis Cluster、、etcd/Raft、 等系统关心 key 到 shard/range/replica group 的定位以及一致性、热点、写放大和失效处理；KVCache 继承 key 定位问题，但 value 的代价函数被推理负载改写。
    

- Mooncake 的核心成本函数： GPU recompute + TTFT + tail latency + cross-node transfer + hit-rate loss + topology mismatch。这已经不是 Redis 或 S3 能直接覆盖的层级。
    

![[Y2026/Q3/0804_KVCache/assets/Image 7.webp|Image]]

02

引言：KVCache ->分布式存储

KVCache 被纳入分布式存储问题空间以后，问题从“key/value 放在哪里”有另外另外一种变形：一个已经消耗 GPU 算力生成的中间态对象，如何在 GPU VRAM、DRAM、SSD、远端对象存储之间流动，并在下一次请求到来前保持可命中、可迁移、可淘汰、可恢复。

Mooncake Store 的价值落在这组对象关系上，而它不是传统意义上的对象存储，也不是简单缓存后端，更接近面向 LLM 推理热链路的短生命周期分布式对象系统：

- master 管理 segment、object、replica、client live state、eviction 和 copy/move/drain task；
    

- storage client 暴露可承载 KVCache 的连续资源区；
    

- Transfer Engine 绕开控制面完成高性能数据搬运。
    

对象仍然是对象，key 仍然需要定位，索引仍然需要承接 replica 与 segment，放置仍然决定下一次命中的成本。变化发生在对象类型、介质层级和成本函数上。

![[Y2026/Q3/0804_KVCache/assets/Image 8.webp|Image]]

03

大规模对象存储经验的迁移入口

大规模对象存储经验比 Redis 缓存经验更能解释 Mooncake 的对象关系。KVCache 同样会被拆成 page、slice、replica、segment、transfer handle 和介质位置。

以海量视频分布式对象存储为例，实时视频通常不会以一个完整大文件落盘，而是按时间切片，例如 30 秒一个时间片；每个时间片再经过 RS 纠删码拆成数据分片和校验分片，分布到不同磁盘节点。读请求到来时，服务侧依赖中心索引定位分片，完成时间片重组，再按业务时间范围对外提供读取。

```
camera_id / channel_id / time_range
-> time slice
-> RS shard set
-> node / disk / offset / length / health state
```

这个系统里最重的工程对象不是“文件”，而是映射关系。元数据中心把分布式磁盘节点上的分片索引合并整理，支撑按时间范围读取视频、缺片恢复，以及迁移任务避开坏盘、慢盘和离线盘。

迁移到 KVCache，映射关系换了一组对象：

```
request / prefix / block key
-> KV cache object
-> replica list
-> segment / endpoint / protocol / memory handle / offload state
```

差异集中在失效成本上，视频对象失效的代价是数据不可读或恢复成本上升；KVCache 失效的代价通常是 miss、prefill 重算、TTFT 上升和 GPU 利用率下降。

04

分布式对象存储的共同骨架

经典分布式对象存储系统的实现形态差异很大，工程问题都会回到同一组对象：对象粒度、索引粒度、放置语义、数据面、故障语义。

|不变量|对象存储里的表现|迁移到 KVCache 后的含义|
|---|---|---|
|对象粒度|文件、block、chunk、time slice、RS shard|KV block、page、tensor slice、replica|
|索引粒度|bucket/key、partition、block id、node/disk/offset|key -> metadata -> replica -> segment|
|放置语义|副本、EC、故障域、冷热层、磁盘组|GPU/DRAM/SSD 层级、拓扑亲和、传输成本|
|控制面边界|tracker、NameServer、MON、ring 不承载热数据搬运|master 做元数据与调度，热数据绕过 master|
|恢复目标|长期可读、完整性、低成本修复|尽快恢复命中率、调度能力和低延迟搬运|

Mooncake 继承的是分布式对象系统的结构压力，目标函数从“长期可靠保存对象”切换成“让计算过的对象在正确位置被再次命中”。

![[Y2026/Q3/0804_KVCache/assets/Image 9.webp|Image]]

05

经典分布式对象存储系统给出的参照坐标

传统对象存储的架构需要过“bucket/key + data node”这一层。系统长期形态由六个切面决定：入口协议如何收敛，元数据如何分片或计算，放置策略如何表达故障域，数据面是否绕开控制面，后台治理如何修复偏差，故障恢复最终恢复的是数据、目录、容量视图还是服务能力。

![[Y2026/Q3/0804_KVCache/assets/Image 10.webp|Image]]

4.1 Ceph：从对象到 PG、OSD、CRUSH 的计算链

Ceph 把 placement 做成可计算规则，而不是中心调度表。RADOS 提供底层对象语义，RGW 提供 S3/Swift 兼容入口，MON 维护集群 map，MGR 承接管理能力，OSD 承载对象、日志、复制、恢复和 scrub。

```
client / RGW
-> MON 获取 cluster map
-> CRUSH(object, pool, rule) 计算 PG 与 OSD set
-> primary OSD 承接读写
-> replica / EC shard 写入其他 OSD
MON/MGR
-> map epoch / OSD state / pool rule / health view
OSD
-> object data / peering / recovery / backfill / scrub
```

Ceph 把对象位置从“中心数据库查表”改成“基于集群视图的确定性计算”。CRUSH 把 host、rack、root、device class、weight、副本数或 EC 规则编码进放置过程。MON 不进入热数据面，OSD 不只是存储进程，还承接副本一致性、peering、recovery、backfill、scrub 等自治工作。

|切面|Ceph 的架构选择|对 Mooncake 的映射|
|---|---|---|
|元数据|MON 维护 cluster map，不保存每个对象的中心位置表|master 维护资源视图，但热链路不能被中心节点拖住|
|放置|CRUSH 用规则表达故障域、拓扑和副本语义|KVCache placement 需要表达 GPU/NUMA/NIC/segment 拓扑|
|数据面|client 直接访问 primary OSD，MON 不转发对象数据|KVCache 应通过 Transfer Engine 在 store client 间搬运|
|后台治理|recovery、backfill、scrub 修复副本、PG 与介质偏差|copy、move、drain、offload 修复可命中、可调度和可搬运状态|

放置策略本身就是系统能力。对象所在位置不只是容量选择，还会把故障域、拓扑、恢复成本和访问链路写进系统行为；而Mooncake 的 KVCache placement 也需要承载这组约束。

4.2 Swift：服务管线、Ring 与后台收敛

Swift 的架构更像一组对象服务pipeline。proxy server 是外部请求入口，ring 把 account/container/object 映射到 partition 和 device，account/container/object server 分别管理命名层级、对象列表和真实对象文件。后台 replicator、auditor、updater、reaper 持续修正系统状态。

```
client
-> proxy server
-> account/container/object ring lookup
-> account server
-> container server
-> object server
background
-> replicator / auditor / updater / reaper
```

Swift 没有把所有读写都压成最短链路，而是用明确的服务分层和后台流程承接大规模对象系统的长期不一致。proxy 承担鉴权、路由、handoff、响应聚合；ring 是分区到设备的映射；account/container 层承接目录与列表语义；object server 承接对象内容。

|切面|Swift 的架构选择|对 Mooncake 的映射|
|---|---|---|
|入口|proxy 是强入口，统一协议、鉴权、容错和聚合|Mooncake master 不能退化成数据 proxy，否则热链路多一跳|
|元数据|account/container/object 三层服务拆开，ring 提供位置映射|KVCache 也要拆开 key、replica、segment、client、task|
|数据面|数据通常经 proxy 到 object server，换取统一服务管线|KVCache 对延迟更敏感，热数据面由 Transfer Engine 承接|
|后台治理|replicator/auditor/updater 是一致性的一部分|eviction、replica clear、drain、offload 也是正确性流程|

对象系统的主负载不止同步请求。对象数量上来以后，清理、修复、审计、迁移都会成为系统主工作负载；映射到 Mooncake，就是 eviction、replica clear、drain、offload 这类动作也属于正确性流程。

4.3 FastDFS：Tracker/Storage 的朴素控制面分离

FastDFS 提供了一组角色对照：tracker 管 storage group、storage server 状态和调度；client 从 tracker 拿到目标 storage 后，直接和 storage 交互；storage group 内部进行副本同步和文件维护。

```
upload:
client -> tracker -> choose group/storage
client -> storage -> file id / metadata
download:
client -> tracker -> locate storage
client -> storage
```

FastDFS 的角色边界清晰：tracker 只管理状态、容量和调度，不转发文件内容；storage 承接文件内容和组内同步；file id 携带定位语义，client 拿到位置后直连数据节点。

这组角色对应到 Mooncake 的 master/storage client：master 像 tracker，storage client 像数据节点。但类比只到入口为止。FastDFS 的 storage 主要承载文件块；Mooncake 的 storage client 暴露的是可被 Transfer Engine 搬运的 segment，segment 还要携带 endpoint、protocol、memory handle、disk/offload state。

|切面|FastDFS 的架构选择|对 Mooncake 的映射|
|---|---|---|
|控制面|tracker 管 storage 状态、容量、分组和调度|master 管 client、segment、object、replica、task 视图|
|数据面|client 拿到 storage 后直连|KVCache 数据绕过 master，走 Transfer Engine|
|放置|group/storage/file id 提供粗粒度定位|key 必须落到 replica/segment/endpoint，而不是只停在 key|
|后台治理|group 内同步和 storage 状态上报|segment drain、copy/move、client remount 属于运行时治理|

4.4 TFS：小对象聚合与底层块治理

TFS 公开资料相对分散，但作为淘宝早期面向海量小文件和图片场景的对象存储实践，它最值得抽取的架构点是 NameServer/DataServer 模型，以及把业务小对象聚合到底层 block 管理。

```
client
-> NameServer 获取 block / DataServer 视图
-> DataServer 读写 block
NameServer
-> block metadata / DataServer membership / replication / balance
DataServer
-> block storage / small object aggregation
```

海量小对象系统要避免对象粒度和物理分配粒度一一对应。小文件如果直接成为底层分配单位，元数据、碎片、随机 IO、迁移和恢复都会被放大。TFS 把业务对象映射到底层 block，再由 NameServer 管理 block 与 DataServer 的关系。

KVCache 面临同一类粒度分离问题：业务粒度、分配粒度、传输粒度、淘汰粒度不能混成一个层次。KV block/page 如果直接等同于底层大块资源，后续会在碎片、元数据和迁移上付出代价。

|切面|TFS 的架构选择|对 Mooncake 的映射|
|---|---|---|
|对象粒度|小文件聚合进 block，避免小对象放大元数据与 IO|KV block/page 需要落到合适的 segment 粒度|
|元数据|NameServer 管 block 与 DataServer 关系|master 管 key、replica、segment、client 关系|
|数据面|DataServer 承接真实 block|storage client 承接真实 segment|
|治理|block 级复制、迁移、均衡|segment 级 copy、move、drain、offload|

4.5 云对象存储：生命周期、治理面与成本产品化

S3、OSS、OBS、Kodo 这类云对象存储虽然不会公开底层所有组件，厂商内部实现也不宜臆测，但是它们对外暴露的能力已经说明系统目标：bucket/key 命名空间、存储类型、生命周期、跨区域复制、版本控制、归档、权限、审计、事件通知、监控和计费。

```
application
-> bucket / key / object API
-> storage class
-> lifecycle transition
-> replication / versioning / archive
-> policy / audit / billing
```

云对象存储把长周期对象的成本函数产品化。用户关心的是 7 天、30 天、半年、一年甚至更长周期里的可靠性、容量成本、跨地域容灾、合规治理和访问费用。厂商不暴露内部放置细节，但存储类型、生命周期、归档取回、跨区复制和权限模型已经把系统优化目标暴露出来。

KVCache 的产品化方向不会复制这套目标。它也需要生命周期、分层、隔离、配额和审计，但第一层压力来自 GPU 重算、TTFT、尾延迟、拓扑搬运和命中率。S3/OSS/OBS/Kodo 更适合作为 checkpoint、冷层、异步 offload 或跨集群共享后端，不适合承担在线推理热链路。

|切面|云对象存储的架构语义|对 Mooncake 的映射|
|---|---|---|
|命名空间|bucket/key、prefix、version、policy|KVCache 仍需要 key，但 key 后面必须有 replica/segment 状态|
|生命周期|标准、低频、归档、深归档之间转储|GPU/DRAM/SSD/S3 也分层，但时间尺度短得多|
|可靠性|多 AZ、跨区域复制、版本控制|KVCache 副本更偏命中率、局部性和重算成本|
|治理|权限、审计、计费、事件、配额|KVCache 生产化后也需要隔离、配额、淘汰策略和证据面|

这些系统并列在一起参照矩阵会比单点类比更清楚。

|系统|控制面|元数据/索引|数据面|放置/恢复|对 Mooncake 最有价值的证据|
|---|---|---|---|---|---|
|Ceph|MON/MGR|cluster map、PG、CRUSH rule|client -> primary OSD -> replica OSD|CRUSH、recovery、backfill、scrub|placement 是可计算能力，控制面退出热数据面|
|Swift|proxy、ring、后台进程|account/container/object 分层索引|proxy -> object server|replicator、auditor、updater|前台请求之外，后台治理是对象系统主负载|
|FastDFS|tracker|group/storage/file id|client -> storage|group sync、storage 状态上报|tracker/storage 是控制面与数据面分离的朴素模型|
|TFS|NameServer|block/DataServer 映射、小对象索引|client -> DataServer|block 复制、迁移、均衡|业务对象粒度与物理分配粒度必须分离|
|云对象存储|托管化控制面|bucket/key、policy、lifecycle、version|厂商内部对象服务|生命周期、归档、跨区域复制|成本函数会塑造系统产品形态|

06

经典 KV 存储与 KVCache 的边界

对象存储参照系能解释 Mooncake 的 object、segment、placement 和 data plane，但它不能覆盖 key-value 这一侧的全部问题。经典 KV 存储长期处理的是 key 到分片、副本组、leader、range、slot、LSM 文件或 raft group 的定位关系。KVCache 分布式存储继承了 key 定位问题，却改变了 value 的工程属性：value 不再只是业务状态或字节串，而是已经消耗 GPU 算力生成的推理中间态。

Redis Cluster、Dynamo/Cassandra、etcd/Raft、RocksDB/LSM 这几类系统可以作为 KV 侧参照。它们的共同问题包括 hash/range 分片、leader/follower 或 quorum 副本、写入确认、热点 key、重平衡、LSM 写放大、range split、lease 与一致性边界。Mooncake KVCache 不直接复制这些机制，但会重新遇到同一组问题：key 如何落到 replica，replica 如何落到 segment，segment 如何落到 client 和 Transfer Engine endpoint，失效后恢复的是命中能力、调度视图，还是可搬运状态。

|维度|经典分布式对象存储|经典分布式 KV 存储|KVCache 分布式存储|
|---|---|---|---|
|基本对象|object、chunk、shard、time slice|key-value pair、slot、range、raft group|prefix block、KV block、tensor slice、replica|
|索引方式|bucket/key -> object metadata|key -> partition / shard / range / leader|key -> replica -> segment -> client|
|放置逻辑|副本、EC、故障域、冷热层|hash/range、leader/follower、副本组|GPU/DRAM/SSD、NUMA/NIC、拓扑和传输代价|
|数据面|client -> object server / OSD|client -> storage node / shard leader|Transfer Engine 在 store client 间搬运|
|可靠性目标|长期可读、完整性、恢复能力|一致性、可用性、写入确认|命中率、低 TTFT、低重算、可迁移|
|失效代价|数据不可读或恢复成本上升|回源、重试、读写失败、重新选主|prefill 重算、GPU 时间损失、尾延迟上升|
|成本函数|容量、带宽、恢复、治理|写放大、复制、一致性、热点|GPU recompute、TTFT、hit rate、topology mismatch|

三方对照可基本达成：Mooncake KVCache 既不是对象存储换一层 API，也不是普通 KV Store 后面挂一个远端缓存。它使用 key 组织访问入口，使用 replica 和 segment 组织资源位置，使用 Transfer Engine 承担热数据面，最后用推理系统的延迟和 GPU 成本重新定义存储收益。

07

Mooncake 的对象模型

Mooncake源码mooncake-store/include/types.h中的Segment是核心资源对象：

```
struct Segment {
UUID id;
std::string name;
uintptr_t base;
size_t size;
std::string te_endpoint;
std::string protocol;
};
```

Mooncake 的 storage node 不是“某台机器”这么粗颗粒度的实体。master 管理segment：带有逻辑名、基地址、大小、Transfer Engine endpoint 和协议的连续资源区。它可以是来自共享内存，也可以挂接到更复杂的介质层，并最终成为 KVCache object replica 的承载空间。

MasterService 的接口也说明Mooncake Store 不是简单 KV map。

|接口族|语义|
|---|---|
|MountSegment / ReMountSegment / UnmountSegment|管理 client 暴露给集群的 segment|
|GetAllSegments / QuerySegments|查询 segment 容量、使用状态和可分配空间|
|PutStart / PutEnd / GetReplicaList|围绕 key 管理对象写入与 replica 生命周期|
|BatchReplicaClear/ Remove / RemoveAll|清理对象和副本|
|CreateCopyTask / CreateMoveTask / CreateDrainJob|显式建模复制、迁移、下线前疏散|
|Ping / ClientStatus::NEED_REMOUNT|维护 master 视角下的 client 活性与 remount 语义|
|NotifyOffloadSuccess / EvictDiskReplica|把 offload 和 disk replica 状态写回控制面|

对象链条展开：

```
key
-> object metadata
-> replica list
-> segment name / memory handle / disk metadata
-> client / TE endpoint / protocol / capacity state
```

这和视频存储里的 time slice -> RS shard -> node/disk/offset 非常接近。差异在于 Mooncake 的物理对象从磁盘偏移变成 memory segment、Transfer Engine endpoint、protocol 和可能的 disk/offload metadata。索引仍然是系统能力的骨架。

![[Y2026/Q3/0804_KVCache/assets/Image 11.webp|Image]]

08

两个元数据面 & 一个热数据面

Mooncake Store 文档明确区分 Master Service 和 Transfer Engine metadata service。Master Service 是独立 RPC 进程，负责 Mooncake Store 的对象、segment、replica、task 等控制面；Transfer Engine 使用自己的 metadata service，例如 HTTP、etcd、Redis，用于 endpoint 和传输发现。

Mooncake 至少有两个元数据面：

```
Mooncake Store control plane
mooncake_master
-> client / segment / object replica / TTL / eviction / task
Transfer Engine metadata plane
http / etcd / redis
-> endpoint / transport / connection discovery
Data plane
store client <-> Transfer Engine <-> store client
```

master 管对象和资源，Transfer Engine 管搬运链路。两者混在一起，故障判断会变得含糊：一次失败到底是对象不可达、segment 不可用、client 失活，还是 transport endpoint 不可达。

Mooncake 的 client 角色也有清晰区分：global_segment_size = 0 时，它是纯 client，只发请求，不贡献全局存储资源；local_buffer_size = 0 时，它作为纯 server，提供存储空间但不执行 Get/Put；standalone store service 则把 real client 独立成资源服务，减轻嵌入推理进程的 client。

热 KVCache 数据在已注册的 client/segment 之间流动，通过 Transfer Engine 选择更短、更合适的搬运链路。CUDA 也属于这条数据面：它不是让 master 做计算，而是让 GPU memory 成为可识别、可注册、可传输、可调度的存储介质。

![[Y2026/Q3/0804_KVCache/assets/Image 12.webp|Image]]

![[Y2026/Q3/0804_KVCache/assets/Image 13.webp|Image]]

09

成本、可靠性与分层

传统对象存储的核心成本大致是：

```
capacity cost
+ durability cost
+ recovery cost
+ bandwidth cost
+ metadata cost
```

KVCache 分布式存储的成本函数变成：

```
GPU recompute cost
+ TTFT
+ tail latency
+ cross-node transfer cost
+ hit-rate loss
+ topology mismatch
```

传统对象存储里的副本或 EC 主要服务长期可靠性；KVCache 的副本更偏性能和可用性。传统对象存储生命周期管理追求容量成本下降；KVCache 的 TTL、eviction 和 offload 会直接影响下一次请求是否重算。传统对象存储的冷层可以接受秒级甚至分钟级恢复；KVCache 热链路没有这个余量。

8.1 介质层级

GPU 进入存储层级以后，数据面不再只是 socket、磁盘和 page cache 的问题。GPU pointer、registered memory、RDMA NIC、NUMA、GPUDirect、NVLink 都会成为存储链路的一部分。

```
传统对象存储
DRAM cache
-> SSD / NVMe
-> HDD / archive / object backend
KVCache 存储
GPU VRAM / HBM
-> DRAM
-> local SSD / NVMe
-> remote object storage / DFS / S3
```

对 KVCache 来说，介质位置本身就是调度语义。

8.2 可靠性语义

KVCache 不是最终业务数据，很多情况下允许重算，所以可靠性无需100%。它的可靠性接近运行时缓存，但一次 miss 的代价远高于普通缓存。

普通缓存 miss 可能只是回源；KVCache miss 可能触发 prefill 重算，消耗 GPU 时间，拉高 TTFT，拖慢同批请求。KVCache 可靠性不是“永不丢失”，而是让活跃对象的 replica 描述正确，让失效 segment 被及时隔离，让 client 失活后有明确 remount 流程，让 copy/move/drain/offload 与对象写入、删除、引用计数之间保持一致。

|源码语义|工程含义|
|---|---|
|ClientStatus::NEED_REMOUNT|首次连接或 ping TTL 过期后，client 必须重新挂载资源视图|
|OBJECT_HAS_REPLICATION_TASK|对象正在复制/迁移时不能被随意改写|
|OBJECT_REPLICA_BUSY|replica 有并发状态，删除、移动、引用需要受控|
|CreateDrainJob|segment 下线前的数据疏散被显式建模为作业|
|NotifyOffloadSuccess|offload 成功后必须把新状态反馈给 master|

这些状态已经越过普通缓存系统的表层功能，进入存储系统的并发控制和故障语义。Mooncake 的难点不在“有没有 key”，而在 key 背后的 replica、segment、client、transport、offload state 是否在高频变化中保持可解释。

10

Redis、S3 与 Mooncake 的位置

Redis、S3、对象存储仍然会出现在 KVCache 系统候选项里，但它们属于不同层级。

Redis 适合控制面状态、开发验证、小容量低延迟缓存。把大量 tensor value 直接放进 Redis，会遇到内存成本、序列化、网络复制和 GPU 数据面脱节。

S3、OSS、OBS、Kodo 适合冷层、checkpoint、跨集群共享、长期落盘和异步 offload。它们保存的是“可以慢一点恢复”的对象，不适合承担在线推理热链路。

Mooncake 的位置更靠近热层和近热层：

```
GPU VRAM / HBM      最热层，避免 D2H/H2D
DRAM                近热层，容量和延迟折中
local SSD / NVMe    offload 层，承接内存压力
S3 / OBS / OSS      冷层、共享层、checkpoint 层
```

把 Redis、S3、Mooncake 放在同一层比较，需要抓住价值问题是：一个 KVCache 对象当前应该处在哪个层级，它的下一次命中价值是多少，把它搬到另一个层级的成本是多少， 存它与搬它的成本和是否大于计算代价。

11

结语

![[Y2026/Q3/0804_KVCache/assets/Image 14.webp|Image]]

Mooncake 与传统对象存储之间的关系，不在于“像不像某个系统”，而在于它承接了分布式存储长期存在的工程问题：对象如何切分，索引如何组织，放置如何表达拓扑和成本，数据面如何避开控制面，后台任务如何修复系统状态，失效以后恢复的到底是什么。

KVCache 站在存储和计算的交界处。它不是普通缓存，而是已经花 GPU 算力生成出来的中间结果。一次命中可以省掉大量 prefill；一次错误淘汰、错误放置或错误搬运，会直接反映在 TTFT、吞吐和 GPU 利用率上。

传统对象存储追求：

```
long-term durability
+ capacity efficiency
+ recoverability
+ protocol compatibility
```

KVCache 分布式存储追求：

```
low recompute
+ low TTFT
+ high hit rate
+ low copy cost
+ topology awareness
```

存算一体在大模型推理里的具体形态，不是泛泛地把存储和计算放在一起，而是让存储系统理解计算中间态的价值、生命周期、介质位置和搬运成本。

成本直觉也要随之改变：从“数据要长期可靠地存在”，转向“计算过的对象要在正确时间、正确位置、以正确代价被再次命中”。

最后，如果你曾经也是一名存储老兵，那么请拿起手中的冲锋枪，去高性能分布式KVCache战场的搏杀吧，那一定属于你，需要你！

12

参考链接

- Ceph Architecture
    
    https://docs.ceph.com/en/latest/architecture/
    

- OpenStack Swift Architectural Overview
    
    https://docs.openstack.org/swift/latest/overview_architecture.html
    

- FastDFS
    
    https://github.com/happyfish100/fastdfs
    

- Amazon S3 Storage Classes
    
    https://aws.amazon.com/s3/storage-classes/
    

- Alibaba Cloud OSS Overview
    
    https://www.alibabacloud.com/help/en/oss/user-guide/oss-overview
    

- Huawei Cloud OBS Resilience
    
    https://support.huaweicloud.com/intl/en-us/productdesc-obs/obs_03_0377.html
    

- Qiniu Kodo storage type
    
    https://developer.qiniu.com/kodo/5906/storage-type
    

- Mooncake 仓库文档
    
    docs/source/design/mooncake-store.md
    
    docs/source/deployment/mooncake-store-deployment-guide.md
    

- Mooncake 源码对象
    
    mooncake-store/include/types.h
    
    mooncake-store/include/master_service.h_
    
    _mooncake-store/include/real_client.h