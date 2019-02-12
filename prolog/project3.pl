:- import member/2 from basics.
:- import append/3 from basics.
:- import sort/3 from basics.
:- import between/3 from basics.
:- import append/2 from basics.
:- import length/2 from basics.

% to get all edges of the role hierarrchy
edge(X,Y) :- 
    rh(X,Y).

% checking whether it has a cycle or not i.e. it is visited or not.
member_list(_,[]):- !.

member_list(X,[H|T]):- 
  X\= H,
  member_list(X,T).

% to check if there is a path exists from A to B
path(A,B) :-
  walk(A,B,[]).

walk(A,B,V) :-       
  edge(A,X) ,        
  not(member(X,V)), 
  (                 
    B = X;                 
    walk(X,B,[A|V])  
  ).              

roleList(X,Y,L):-
  rh(X,Y).

% this function checks the roles hierarrachy of the given roles

roleList(X,Y,L):-
  rh(X,Z),
  member_list(Z,L),
  roleList(Z,Y,[Z|L]).

% function to get the last item from the given list

lastitem([OnlyOne], OnlyOne).
lastitem([First | Rest], Last) :-
  lastitem(Rest, Last).

%this function calls the rolelist function to get the roles from the hierarrchy.

descended_roles(X,Y,L):-
  ur(X,Y).

descended_roles(X,Y,L):-
  ur(X,Z),
  member_list(Z,L),
  roleList(Z,Y,[Z|L]).

%function to compute roles of a given user.

authorized_roles(User,Roles):-
  findall(V,descended_roles(User,V,[]),Res),
  sort(Res,Roles).

% to get permissions for the User
authorized_permissions(User,PermList) :-
    authorized_roles(User,Roles),
    setof(P, X^Y^(member(X, Roles), rp(X,Y), P is Y), PermList).

set_of_permissions(User,PermList) :-
    authorized_roles(User,Roles),
    setof(P, X^Y^(member(X, Roles), rp(X,Y), P is Y), UnsortedPerm),
    sort(UnsortedPerm, PermList).

%base condition for breaking the recursion.
getpermissionlist(1, L):-
    set_of_permissions(1, L),!.

% This function iterates for all users
getpermissionlist(X, L):-
      set_of_permissions(X, L);
      M is X-1,
      getpermissionlist(M, L).
      
% to get minroles for all users
minRoles(S):-
    users(X),
    setof(L, getpermissionlist(X,L), ResPermList),
    length(ResPermList, S).