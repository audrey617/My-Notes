Lesson1 & 1.5 Introduction to Software Analysis </br>

Undecidability: </br>
A decision problem is a problem that requires a yes or no answer. A decision problem that admits no algorithmic solution is said to be undecidable [link](https://www.cs.rochester.edu/u/nelson/courses/csc_173/computability/undecidable.html#:~:text=Definition%3A%20A%20decision%20problem%20is,is%20said%20to%20be%20undecidable.) [wiki](https://en.wikipedia.org/wiki/Undecidable_problem)</br>


Soundness and Completeness</br>

Dynamic analysis - unsound (may miss error, false negative)</br>
Static analysis - incomplete (may report spurious errors, false positive)</br>

A sound analysis - we can trust the correctness of the output. Soundness: An analysis True is definitely true in program but not all True may be proven. It can say some true programs are false. trivialSoundAnalysis() returns false for all inputs</br>
A complete analysis - we can trust the incorrectness of the ouput. Completeness:  A analysis false is definitely false in program but not all false may be proven. It can say some false program to true. trivialCompleteAnalysis() returns true for all inputs</br>

Soundness and completeness are relative terms and depends on what we are looking for. Are we analyzing correct programs or error-containing programs.</br>
Correct programs: All sound program are correct program. all correct program are complete. </br>
Error-containing programs: All complete programs are error-containing program, all error-containing are sound program.</br>
Frame of reference: correct software -> sound analysis; error-containing programs -> complete analysis</br>


