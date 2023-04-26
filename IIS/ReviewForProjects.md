My reviews and helpful reference links. It is pretty easy course. I skipped the credit project and still got more than 100% in total (since binary exploitation has extra points)

</br>**Project1 - Man in the middle** </br>
This project somehow reminds me of the steam game Hacknet. The working process on this project is a bit suffering and kinda like a puzzle. I got stuck on part 2 for a long while. But overall it is fun. I finished the project in 6 hours, while few students completed the lab in 3 hours. I assume they are very good at wireshark and have some security background before the lab. </br>

The below references help me unstuck. I wish I could spot some links earlier </br>
Example IRC Communications http://chi.cs.uchicago.edu/chirc/irc_examples.html</br>
Wireshark key concepts Display filters: https://wiki.wireshark.org/DisplayFilters </br>
Extract files from PCAP using Export https://hackertarget.com/wireshark-tutorial-and-cheat-sheet/</br>
understand DCC SEND https://en.wikipedia.org/wiki/Direct_Client-to-Client#DCC_SEND</br>
Cyberchef https://gchq.github.io/CyberChef/</br>


</br>**Project2 - Malware Analysis** </br>
This project can be better. It is cool to analyze the Joesandbox malware reports in part 1 and understand what behaviors different malwares result in. However, the part 2 is to read an old paper and twist some ML config. The approach doesn't look practical nor interesting. I did a quick scan of reports & paper and finished in 4 hours. However, if you only care about passing gradescope, you just need to be smart on guessing answers and can finish the entire project in less than 30 mins. </br>

Reference</br>
Mirai botnet: https://www.cloudflare.com/learning/ddos/glossary/mirai-botnet/</br>
RegAsm virus: https://malwaretips.com/blogs/remove-regasm-exe/#:~:text=The%20Regasm.exe%20process%20is%20part%20of%20a%20Trojan%20Horse,itself%20as%20a%20legitimate%20process.</br>


</br>**Project3 - Web Security** </br>
This project is about XSRF, XSS and SQL injection to a provided unsecured website. I never thought about these attacks and learnt a lot. Overall it is a good assignment. However, I do feel a lot of pain due to no previous experience on Javascript. In addition, tests are not available, which force me to check details on VM again and again.  It took me multiple nights to finish this project. </br>

Reference:</br>
XSS https://medium.com/@R4v3ncl4w/step-20-cross-site-scripting-xss-1df10ff7fd12</br>
SQL injection https://en.wikipedia.org/wiki/SQL_injection</br>
(In Chinese) How to prevent XSS attack: https://tech.meituan.com/2018/09/27/fe-security.html</br>


</br>**Project4 - ML on CLAMP** </br>
This homework is a big distraction from Security itself and every task is basic ML implementation. I think it is one of the worst assignments I have had in this program: The quality of this assignment document is bad, some function descriptions are very unclear, and no local tests are provided for the majority parts. In addition, the GS tests usually provide no feedback on failed tests and the non-public test design is pretty silly; one design example is that One Hot Encoding value has to be 0.0/1.0 float format and GS test will fail silently if your value is 0 or 1. Even I am pretty familiar with all concepts and relevant coding, I spent 11.5 hours to pass all GS tests. My workload should reduce by at least half if the project is designed better. Let alone I didn't learn any security concepts from this project.</br>

Reference:</br>
args and kwargs in Python: https://www.geeksforgeeks.org/args-kwargs-python/

</br>**Project5 - Cryptography (RSA)** </br>
Much more interesting project. Different attacks on RSA. Total 11.5 hours.

Reference:</br>
Modular exponentiation https://en.wikipedia.org/wiki/Modular_exponentiation#Memory-efficient_method</br>
The Mathematics behind RSA http://www.cs.sjsu.edu/~stamp/CS265/SecurityEngineering/chapter5_SE/RSAmath.html</br>
Understand Python Decimal</br>

</br>**Project6 - Binary Exploitation** </br>
This binary exploitation project is the best project so far in this IIS course. It blows my mind on how to abuse the unsafe code. This project covers assembly, buffer overflow, reverse engineer and ROP (return oriented programming). Super cool! Brute-force can work in part 1 and part 2 tasks once you spot some patterns. However, part 3 requires more research - make sure use GDB for exploration and read project doc carefully. Total 12 hours.

Reference:</br>
pwntools document https://docs.pwntools.com/en/stable/util/cyclic.html </br>
GDB cheatsheet https://darkdust.net/files/GDB%20Cheat%20Sheet.pdf</br>
Linux strings command https://www.javatpoint.com/linux-strings-command</br>
Return Oriented Programming https://ctf101.org/binary-exploitation/return-oriented-programming/</br>
ROP tutorial https://www.youtube.com/watch?v=8zRoMAkGYQE&list=PLchBW5mYosh_F38onTyuhMTt2WGfY-yr7&index=13</br>


</br>**Project7 - Log4Shell** </br>
This project covers https://www.lunasec.io/docs/blog/log4j-zero-day/ issue and has the hardest puzzle to think about in this IIS course. Finished the project in 10 hours but it is very stressful. Suggestion is to look at reference material when get stuck. If you has issue on finishing flag 4, then work on flag 5 and 6 first. The knowledge leanred from flag 5/6 will help sort out flag 4. Generally it's still very interesting. No extra references are needed. Reference Material in lab document covers everything. READ ALL!

