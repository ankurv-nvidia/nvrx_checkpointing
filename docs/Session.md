Communication Channels

1. Messages + Continuations
* Packetized Messages
* Init Message-headers with data
* Continuation Message with data
* Each packet has the context headers 

    Shared Memory (Single-thread-ctrl, Mult-threaded-data)
        * Thrd1:Adjust-MTU(Default)
        * Thrd1:Start(Foo, MsgSize)
        * Thrd1:Data(Foo, Range)
        * Thrd2:Data(Foo, Range)
        * Thrd1:End(Foo)
    Sockets + Buffers
        * /start

Pros: 
* easy multi-threading
* server coordination is simple


2. Ctrl-Channel + Data-Channel
* Headers in ctrl describe what to expect from data channel
* Each ctrl-message may or maynot have a data-packet
* Data-Channel has zero context


Pros:
* Better memory management and buffer management

Cons:
* Sequencing of multithreaded operations is painful
    * A single stream (ctrl+data) has to be sequential

Open-Questions
* How to implement Ctrl-Data channels with WebServers
    * Url + headers
* How to multi-plex multiple parallel requests without creating more endpoints


Medium
1. Shared Memory
2. Sockets (single-port)
3. Infiniband RDMA (2-sided)

1. Shared Memory

* Avoid 2-way comms and sync 
* Single Ctrl Stream
    * Messages can be queued by anyone anytime
    * Message contain the stream for the data-packet for the ctrl-packat
* Multiple Data Streams
    * 

Shared Memory
1. Ctrl + Data
W