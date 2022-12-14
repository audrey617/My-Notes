Some topics I'm interested in or just some thoughts. Outside of the lecture/exam scope.</br>

**More about Memcached** </br>
Hardware requirement for Memcached https://github.com/memcached/memcached/wiki/Hardware</br>
Scaling memcached at Facebook https://engineering.fb.com/2008/12/12/core-data/scaling-memcached-at-facebook/</br>

**Chord - Is chord really correct?** </br>
1. https://www.cs.princeton.edu/courses/archive/fall18/cos561/docs/ChordClass.pdf </br>
2. https://arxiv.org/pdf/1502.06461.pdf </br>


**RDMA Design** </br>
1. Ziegler, Tobias, Viktor Leis, and Carsten Binnig. "Rdma communciation patterns." Datenbank-Spektrum 20.3 (2020): 199-210. https://link.springer.com/article/10.1007/s13222-020-00355-7
2. Reda, Waleed, et al. "RDMA is Turing complete, we just did not know it yet!." 19th USENIX Symposium on Networked Systems Design and Implementation (NSDI 22). 2022. https://wreda.github.io/papers/redn-nsdi22.pdf </br>


**Spanner vs BigQuery** </br>
This was the question bothered me for a while as these two products look very similar to me. I previously used BQ a lot at work but was never exposed to Spanner. Both BQ and Spanner are built on Colossus and on Borg, can scale up well, and use relational DBMS as primary database model. Now I think what actually distinguish them is one is OLAP data warehouse and the other is OLTP database, which can also explain how them are used together. Happy to discuss more</br>

https://cloud.google.com/blog/topics/developers-practitioners/how-spanner-and-bigquery-work-together-handle-transactional-and-analytical-workloads</br>
https://aws.amazon.com/data-warehouse/</br>
https://stackoverflow.com/questions/21900185/what-are-oltp-and-olap-what-is-the-difference-between-them</br>
https://db-engines.com/en/system/Google+BigQuery%3BGoogle+Cloud+Spanner</br>
https://hevodata.com/learn/spanner-vs-bigquery/#1 </br>


**More about Transaction** </br>
Unfortunately the CS6400 Database course focuses too much on SQL & basic design applicaiton but has little depth of or completely skipped advanced system-level topics including transaction. For these concepts, good to read the DB textbook fundamentals of database systems Part 9 transaction and Part 10 distributed database for more information. Peter Bailis's blog is also helpful.</br>

"Serializability is a sufficient condition for ACID consistency but it is not strictly necessary. A database with serializability (“I” in ACID), provides arbitrary read/write transactions and guarantees consistency (“C” in ACID)", "For distributed systems nerds: achieving linearizability for reads and writes is, in a formal sense, “easier” to achieve than serializability. here’s some intuition: terminating atomic register read/write operations are achievable in a fail-stop model. Yet atomic commitment—which is needed to execute multi-site serializable transactions (think: AC is to 2PC as consensus is to Paxos)—is not: the FLP result says consensus is unachievable in a fail-stop model (hence with One Faulty Process), and (non-blocking) atomic commitment is “harder” than consensus (see also). Also, keep in mind that linearizability for read-modify-write is harder than linearizable read/write. " - Peter Bailis</br>
[When is "ACID" ACID? Rarely](http://www.bailis.org/blog/when-is-acid-acid-rarely/)</br>
[Without Conflicts, Serializability Is Free](http://www.bailis.org/blog/without-conflicts-serializability-is-free/)</br>
[Linearizability versus Serializability](http://www.bailis.org/blog/linearizability-versus-serializability/)</br>
[Understanding Weak Isolation Is a Serious Problem](http://www.bailis.org/blog/understanding-weak-isolation-is-a-serious-problem/)</br>
More posts from Peter Bailis blog http://www.bailis.org/blog/</br>



**Papers and articles from James Mickens** </br>
Woud love to read more articles from him!</br>
The Saddest Moment is here and more: https://mickens.seas.harvard.edu/wisdom-james-mickens</br>
More research papers: https://mickens.seas.harvard.edu/publications?page=1</br>
