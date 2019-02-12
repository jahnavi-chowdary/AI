:- consult(project3).

% input format
users(5).
roles(5).
perms(5).

ur(1,2).
ur(2,3).
ur(3,4).
ur(4,4).
ur(2,4).

rp(1,2).
rp(2,4).
rp(3,5).
rp(4,2).
rp(5,2).

rh(1,3).
rh(2,3).
rh(3,5).
rh(4,5).
rh(5,4).
rh(5,3).

% output format
% authorized_roles(2, List_roles).
% List_roles = [3,4,5]

% authorized_permissions(1, List_Permissions).
% List_Permissions = [2,4,5]

% min_Roles(S).
% S = 2