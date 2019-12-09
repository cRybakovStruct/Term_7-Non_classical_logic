fact(N, F) :-
( N=0, F=1;
N>0,
N1 is N-1,
fact(N1, F1),
F is N*F1). 


?-
N=4,
fact(N, F), write(N), write('! = '), write(F), nl.



/*?-
write("Hello World"), nl.*/