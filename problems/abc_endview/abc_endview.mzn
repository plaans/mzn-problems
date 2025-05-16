% 
% ABC End View puzzle in MiniZinc.
% 
% This is also knows as "Easy as ABC" and "Last Man Standing".
% 
% From Fun With Puzzles: "ABC End View: A1"
% http://www.funwithpuzzles.com/2009/12/abcd-end-view-a1.html
% """
% This the classical puzzle and appeared in many World Puzzle Championships.
% 
% This puzzle is also known with following names
% 1. Easy as ABC
% 
% "ABC End View" Puzzle Instructions:
% Enter the letters ABC such that each letter is exactly once, in all of the 
% rows and columns. One cells will remain empty in each row and column. 
% The letters outside the grid show which letter you come across first 
% from that direction.
% 
%       ABC End View
% 
%            A               
%        _ _ _ _
%      C _ _ _ _
%      C _ _ _ _ B
%        _ _ _ _
%          B
% """
%
% Note: There are some problem instances below that use A..D and
%       5x5 or 6x6 grid which means that there may be 2 empty cells
%       before/after.
%
%
% Also see:
%   http://www.janko.at/Raetsel/AbcEndView/index.htm
%
% The diagonal constraint means that the two diagonal should
% also be alldifferent (except empty cells). See below 
% for an example (problem 168) from 
% http://www.janko.at/Raetsel/AbcEndView/168.a.htm
%

% 
% This MiniZinc model was created by Hakan Kjellerstrand, hakank@bonetmail.com
% See also my MiniZinc page: http://www.hakank.org/minizinc/
%

include "globals.mzn"; 

int: n;
int: max_letter;
int: dist = n-max_letter+1; % number of accepted empty cells before/after
bool: diagonal; % has diagonal constraint?

% hints
array[1..n] of int: row_upper;
array[1..n] of int: row_lower;

array[1..n] of int: col_left;
array[1..n] of int: col_right;

% For problem instance
int: A = 1;
int: B = 2;
int: C = 3;
int: D = 4;
int: E = 5;
int: F = 6;

% Presentation
array[0..6] of string: str = array1d(0..6, ["_", "A","B","C","D","E","F"]);

% For global cardinality
array[0..max_letter] of int: gcc = array1d(0..max_letter, [i | i in 0..max_letter] );
array[0..max_letter] of int: counts = array1d(0..max_letter, [n-max_letter] ++ [1 | i in 1..max_letter] );


% decision variables
array[1..n, 1..n] of var 0..max_letter: x;

solve satisfy;

constraint
  trace("gcc: " ++ show(gcc) ++ "\n" ++
        "counts: " ++ show(counts) ++ "\n" ++
        "dist: " ++ show(dist) ++ "\n", 1=1)
;

%
% a start with letter c and accept d empty slots.
% 
predicate start_with(array[int] of var int: a, int: c, int: d) =
   if c > 0 then
     exists(i in 1..d) (
       a[i] = c 
       /\       
       forall(j in 1..i-1) (
          a[j] = 0
       )
     )
   else 
     true
   endif
;


%
% a ends with letter c and accept d empty slots
%
predicate end_with(array[int] of var int: a, int: c, int: d) =
   let {
     int: len = length(a)
   } in
   if c > 0 then
      exists(i in n-d..n) (
        a[i] = c 
        /\       
        forall(j in i+1..n) (
          a[j] = 0
        )
     )
   else
      true
   endif
;


%
% Latin square except 0
%
constraint
   forall(i in 1..n) (
      global_cardinality([x[i,j] | j in 1..n], gcc, counts) 
   )
   /\
   forall(j in 1..n) (
      global_cardinality([x[i,j] | i in 1..n], gcc, counts) 
   )

   /\
   if diagonal then 
     global_cardinality([x[i,i] | i in 1..n], gcc, counts) 
     /\
     global_cardinality([x[i,n-i+1] | i in 1..n], gcc, counts) 
   else 
     true
   endif
;

%
% The hints
%
constraint
  forall(j in 1..n) (
     let {
       array[1..n] of var 0..n-1: tmp = [x[i,j] | i in 1..n]
     } in
     start_with(tmp, row_upper[j], dist)
     /\
     end_with(tmp, row_lower[j], dist)
  )
  /\
  forall(i in 1..n) (
     let {
       array[1..n] of var 0..n-1: tmp = [x[i,j] | j in 1..n]
     } in
     start_with(tmp, col_left[i], dist)
     /\
     end_with(tmp, col_right[i], dist)
  ) 
;


output 
[
   if j = 1 then "\n" else " "  endif ++
     show(str[fix(x[i,j])])
   | i, j in 1..n
]
++ ["\n"]
;