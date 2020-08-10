import sys
import copy
import os.path
import itertools


global G
G = [[0],
     [1],
     [1],
     [1, 2],
     [1, 3],
     [1, 2, 3, 4],
     [1, 5],
     [1, 2, 3, 4, 5, 6],
     [1, 3, 5, 7],
     [1, 2, 4, 5, 7, 8],
     [1, 3, 7, 9],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     [1, 5, 7, 11],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
     [1, 3, 5, 9, 11, 13],
     [1, 2, 4, 7, 8, 11, 13, 14],
     [1, 3, 5, 7, 9, 11, 13, 15],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
     [1, 5, 7, 11, 13, 17],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
     [1, 3, 7, 9, 11, 13, 17, 19]]

global P
P = list()

global donu
donu = [0, 1, 2, 3, 4, 5]

def new_matrix(c, k, p):
    M = []
    row = [0 for i in range(0, k * c + 2, 1)]
    for i in range(0, k * c + 2, 1):
        M.append(copy.deepcopy(row))
    for i in range(0, k * c, 1):
        M[1][i + 2] = divmod(i, k)[0]
        M[i + 2][1] = int(divmod(divmod(i, k)[0] + p, c)[1])
    return M


def doplnok(M, c2, k2, c3, k3, p, d_row):
    poct = list()
    dop = list()
    for i in range(0, c2, 1):
        poct.append([0 for i in range(0, c3, 1)])
        dop.append([])
    for i in range(1, d_row, 1):
        a = divmod(M[d_row][i], c2)[1]
        b = divmod(M[d_row][i] - a, c2)[0]
        poct[a][b] = poct[a][b] + 1
    for i in range(0, c2, 1):
        for j in range(0, c3, 1):
            for l in range(poct[i][j], divmod(k3, c2)[0], 1):
                dop[i].append(j)
    d = copy.deepcopy(dop)
    return d


def bloks(M, c2, k2, c3, k3, p ,d_row):
    c = c2*c3
    r = [[0,i] for i in range(0, c2*k2 + 2, 1)]
    nr = [[0,i] for i in range(0, c2*k2 + 2, 1)]
    for i in range(1, d_row):
        for j in range(d_row + 1, c2*k2 + 2, 1):
            r[j][0] = r[j][0] * c + M[i][j]
            nr[j][0] = nr[j][0] * c + divmod(M[i][j], c2)[1]
    n = list()
    poz = list()
    for i in range(0, c2, 1):
        n.append(0)
        poz.append([])
    nr[d_row + 1:] = sorted(nr[d_row + 1:])
    r[d_row + 1:] = sorted(r[d_row + 1:])
    nblk2 = [[0],[nr[d_row + 1][1]]]
    nblk6 = [[0],[r[d_row + 1][1]]]
    for i in range(d_row + 1, c2*k2 + 1, 1):
        if nr[i][0] != nr[i + 1][0]:
            nblk2.append([nr[i + 1][1]])
        else:
            nblk2[-1].append(nr[i + 1][1])
        if r[i][0] != r[i + 1][0]:
            nblk6.append([r[i + 1][1]])
        else:
            nblk6[-1].append(r[i + 1][1])
        poz[M[d_row][i]].append(i)
    poz[M[d_row][c2*k2 + 1]].append(c2*k2 + 1)
    nblk2 = sorted(nblk2)
    nblk6 = sorted(nblk6)
    nblk2.append([c2*k2 + 2])
    nblk6.append([c2 * k2 + 2])
    return (nblk2, nblk6, poz)


def bl(c2, nadb6, d2, d_row):
    bloky = [[0] for i in range(0, c2, 1)]
    poc = [0 for i in range(0, c2, 1)]
    pb = 2
    pos = -1
    for i, j in enumerate(nadb6[1:-1]):
        for k in range(0, c2, 1):
            l = d2[j[0]:nadb6[2 + i][0]].count(k)
            poc[k] = poc[k] + l
            if l != 0:
                bloky[k].append(poc[k])
    return bloky


def lex_post_bloky(bloky, doplnok):
    b = bloky
    d = doplnok
    for i in range(len(b)-3, -2, -1): # lebo s prvým blokom sa to hrať nemá, ten je daný
        if i == -1:
            return 0
        pivot = 0
        if d[b[i+2]-1]>d[b[i]]:
            for j in range(b[i + 1] - 1, b[i] - 2, - 1):
                if d[j] < d[b[i + 2] - 1]:
                    sufix = d[j:b[i+1]]
                    len_sufix = b[i+1] - j
                    pivot = j
                    break
            Sufix = list()
            for j in range(len(b)-2,i,-1):
                Sufix = Sufix + d[b[j]:b[j+1]]
            len_Sufix = len(Sufix)
            j = 0
            k = 0
            J = 0
            d1 = list()
            d2 = list()
            while j < len_sufix:
                if k < len_Sufix:
                    if Sufix[k]<=sufix[j]:
                        d2.append(Sufix[k])
                        k = k + 1
                    else:
                        d2.append(sufix[j])
                        j = j + 1
                        if j == 1:
                            J = len(d2)
                else:
                    d2 = d2 + sufix[j:]
                    break
            d2 = d2 + Sufix[k:]
            d1 = d2[J:J + len_sufix]
            d2 = d2[:J] + d2[J + len_sufix:]
            return d[0:pivot] + d1 + d2


def lex_post_pref(bloky, doplnok):
    b = bloky[:3] + [bloky[-1]]
    d = doplnok
    d2 = sorted(d[b[2]:])
    d = d[:b[2]] + d2
    t = 0
    tt = 0
    for i in range(b[1], b[2], 1):
        t = t + d[i]
        if d[i] != 1:
            tt = tt + 1
    while t > 3*tt:
        for i in range(len(b)-3, -1, -1): # lebo s prvým blokom sa to hrať nemá, ten je daný
            if i == 0:
                return 0
            pivot = 0
            if d[b[i+2]-1]>d[b[i]]:
                for j in range(b[i + 1] - 1, b[i] - 2, - 1):
                    if d[j] < d[b[i + 2] - 1]:
                        sufix = d[j:b[i+1]]
                        len_sufix = b[i+1] - j
                        pivot = j
                        break
                Sufix = list()
                for j in range(len(b)-2,i,-1):
                    Sufix = Sufix + d[b[j]:b[j+1]]
                len_Sufix = len(Sufix)
                j = 0
                k = 0
                J = 0
                d1 = list()
                d2 = list()
                while j < len_sufix:
                    if k < len_Sufix:
                        if Sufix[k]<=sufix[j]:
                            d2.append(Sufix[k])
                            k = k + 1
                        else:
                            d2.append(sufix[j])
                            j = j + 1
                            if j == 1:
                                J = len(d2)
                    else:
                        d2 = d2 + sufix[j:]
                        break
                d2 = d2 + Sufix[k:]
                d1 = d2[J:J + len_sufix]
                d2 = d2[:J] + d2[J + len_sufix:]
                d = d[0:pivot] + d1 + d2
                t = 0
                tt = 0
                for i in range(b[1], b[2], 1):
                    t = t + d[i]
                    if d[i] != 1:
                        tt = tt + 1

    print(d)
    return d


def lex_post_bloky_s(bloky, doplnok):
    b = bloky
    d = doplnok
    q = 0
    for i in range(len(b)-3, -1, -1): # lebo s prvým blokom sa to hrať nemá, ten je daný
        if i == 0:
            return 0
        pivot = 0
        if i < 3:
            print(d)
        if d[b[i+2]-1]>d[b[i]]:
            for j in range(b[i + 1] - 1, b[i] - 2, - 1):
                if d[j] < d[b[i + 2] - 1]:
                    sufix = d[j:b[i+1]]
                    len_sufix = b[i+1] - j
                    pivot = j
                    break
            Sufix = list()
            for j in range(len(b)-2,i,-1):
                Sufix = Sufix + d[b[j]:b[j+1]]
            len_Sufix = len(Sufix)
            j = 0
            k = 0
            J = 0
            d1 = list()
            d2 = list()
            while j < len_sufix:
                if k < len_Sufix:
                    if Sufix[k]<=sufix[j]:
                        d2.append(Sufix[k])
                        k = k + 1
                    else:
                        d2.append(sufix[j])
                        j = j + 1
                        if j == 1:
                            J = len(d2)
                else:
                    d2 = d2 + sufix[j:]
                    break
            d2 = d2 + Sufix[k:]
            d1 = d2[J:J + len_sufix]
            d2 = d2[:J] + d2[J + len_sufix:]
            d = d[0:pivot] + d1 + d2
            t = 0
            tt = 0
            for i in range(b[1],b[2],1):
                t = t + d[i]
                if d[i] != 1:
                    tt = tt + 1
            if t <= 3*tt:
                return d
            else:
                print(d)
                d = lex_post_pref(bloky, d)
                print(d)
                return d


def lex_post_komb(bloky, doplnok):
    b = bloky
    d = doplnok
    for i in range(len(b)-3, -2, -1):
        pivot = 0
        if i == -1:
            return 0
        if d[b[i+2]-1] > d[b[i]]:
            for j in range(b[i + 1] - 1, b[i] - 2, - 1):
                if d[j] < d[b[i + 2] - 1]:
                    sufix = d[j:b[i+1]]
                    len_sufix = b[i+1] - j
                    pivot = j
                    break
            Sufix = list()
            for j in range(len(b)-2,i,-1):
                Sufix = Sufix + d[b[j]:b[j+1]]
            len_Sufix = len(Sufix)
            j = 0
            k = 0
            J = 0
            d1 = list()
            d2 = list()
            while j < len_sufix:
                if k < len_Sufix:
                    if Sufix[k]<=sufix[j]:
                        d2.append(Sufix[k])
                        k = k + 1
                    else:
                        d2.append(sufix[j])
                        j = j + 1
                        if j == 1:
                            J = len(d2)
                else:
                    d2 = d2 + sufix[j:]
                    break
            d2 = d2 + Sufix[k:]
            d1 = d2[J:J + len_sufix]
            d2 = d2[:J] + d2[J + len_sufix:]
            return d[0:pivot] + d1 + d2


def lex_post_nadbloky(c2, nadb2, nadb6, D2, per):
    d2 = [D2[i] for i in per]
    for i in range(len(nadb2) - 2, -2, -1):
        if i == 0:
            return 0
        d2_d = copy.deepcopy([d2[j] for j in nadb2[i]])
        l = 0
        bl = [0]
        for j, k in enumerate(nadb6):
            if k[0] in nadb2[i]:
                l = l + len(k)
                bl.append(l)
        d2_d = lex_post_komb(bl, d2_d)
        if d2_d != 0:
            d2_f = copy.deepcopy(d2)
            for k, j in enumerate(nadb2[i]):
                d2_f[j] = d2_d[k]
            for j in range(i + 1, len(nadb2) - 1, 1):
                for k, l in enumerate(nadb2[j]):
                    d2_f[l] = D2[l]
            perm = [j for j in range(nadb6[-1][0])]
            for j in nadb2[1:-1]:
                bl = [[] for i in range(0, c2, 1)]
                for k, l in enumerate(j):
                    bl[D2[l]].append(l)
                poc = [0 for i in range(0, c2, 1)]
                for k, l in enumerate(j):
                    perm[l] = bl[d2_f[l]][poc[d2_f[l]]]
                    poc[d2_f[l]] = poc[d2_f[l]] + 1
            return perm


def lex_post(x):
    zac = 0
    for i in range(len(x) - 1, -1, -1):
        if x[i] > x[i - 1]:
            zac = i
            break
    if zac == 0:
        return 0
    else:
        prvy = x[zac - 1]
        for i in range(zac, len(x), 1):
            if x[i] > prvy:
                nahr = i
        nahr_value = x[nahr]
        x[nahr] = prvy
        x[zac - 1] = nahr_value
        y = copy.deepcopy(x)
        for i in range(0, len(x) - zac, 1):
            y[i + zac] = x[len(x) - 1 - i]
        return y


def symetrizuj(M, c, c2, k, p, d_row, pivot):
    d = c
    for j in range(1, d_row + 1, 1):
        if j > pivot:
            d = c2
        for i in range(0, c*k + 2, 1):
            if j < i:
                M[i][j] = divmod(M[j][i] + p, d)[1]
    return M


def check(c, k, p, M, d_row):
    for i in range(0, d_row, 1):
        t = 1
        rozdiel = [0 for i in range(0, c, 1)]
        for j in range(0, c*k + 2, 1):
            if i != j and d_row != j:
                rozdiel[divmod(M[i][j]-M[d_row][j], c)[1]] = rozdiel[divmod(M[i][j]-M[d_row][j], c)[1]] + 1
        for j in range(0, c, 1):
            if rozdiel[j] != k:
                t = 0
        if t == 0:
            break
    return t


def precheck(c, k, p, M, d_row, pivot):
    for i in range(0, d_row, 1):
        t = 1
        rozdiel = [0 for i in range(0, c, 1)]
        for j in range(0, pivot, 1):
            if i != j and d_row != j:
                rozdiel[divmod(M[i][j]-M[d_row][j], c)[1]] = rozdiel[divmod(M[i][j]-M[d_row][j], c)[1]] + 1
        for j in range(0, c, 1):
            if rozdiel[j] > k:
                t = 0
        if t == 0:
            break
    return t


def num(c, arr):
    n = 0
    if c == 6:
        for i in arr:
            n = n*c + donu[i]
        return n
    else:
        for i in arr:
            n = n*c + i
        return n


def num_matrix(c, k, M, d_row):
    a = 0
    if c == 6:
        for j in range(1, d_row + 1, 1):
            for i in range(2, c*k + 2, 1):
                if j < i:
                    a = a * c + donu[M[j][i]]
        return a
    else:
        for j in range(1, d_row + 1, 1):
            for i in range(2, c * k + 2, 1):
                if j < i:
                    a = a * c + M[j][i]
        return a


def denum(c, dlzka, num):
    arr = [0 for i in range(0, dlzka, 1)]
    if c == 6:
        for i in range(dlzka - 1, -1, -1):
            arr[i] = donu[divmod(num, c)[1]]
            num = divmod(num, c)[0]
        return arr
    else:
        for i in range(dlzka - 1, -1, -1):
            num, arr[i] = divmod(num, c)
        return arr


def denum_matrix(c, c2, k, p, num, d_row, pivot):
    m = [0 for i in range(0, c*k + 2, 1)]
    M = [copy.deepcopy(m) for i in range(0, c*k + 2, 1)]
    if c == 6:
        for j in range(d_row, 0, -1):
            for i in range(c * k + 1, 1, -1):
                if j < i:
                    num, b = divmod(num, c)
                    M[j][i] = donu[b]
        return symetrizuj(M, c, c2, k, p, d_row, pivot)
    else:
        for j in range(d_row, 0, -1):
            for i in range(c * k + 1, 1, -1):
                if j < i:
                    num, M[j][i] = divmod(num, c)
        return symetrizuj(M, c, c2, k, p, d_row, pivot)


def usp_nor_tvar(M, c, c2, k, p, perm, g, d_row, pivot):
    poradie = [[i for i in range(0, c*k + 2, 1)], [0 for i in range(0, c*k + 2, 1)]]
    nula = perm[0]
    if c == 6:
        for i in range(1, c*k + 2, 1):
            for j in range(0, d_row + 1, 1):
                if j < i:
                    poradie[1][i] = poradie[1][i] * c + donu[divmod((M[perm[j]][perm[i]] - M[nula][perm[i]] - M[perm[j]][nula])*g, c)[1]]
                else:
                    poradie[1][i] = poradie[1][i] * c
    else:
        for i in range(1, c * k + 2, 1):
            for j in range(0, d_row + 1, 1):
                if j < i:
                    poradie[1][i] = poradie[1][i] * c + divmod((M[perm[j]][perm[i]] - M[nula][perm[i]] - M[perm[j]][nula]) * g, c)[1]
                else:
                    poradie[1][i] = poradie[1][i] * c
    t = 0
    while t == 0:
        t = 1
        for i in range(0, c*k + 1, 1):
            if poradie[1][i] > poradie[1][i+1]:
                poradie[0][i] = poradie[0][i] ^ poradie[0][i + 1]
                poradie[1][i] = poradie[1][i] ^ poradie[1][i + 1]
                poradie[0][i + 1] = poradie[0][i] ^ poradie[0][i + 1]
                poradie[1][i + 1] = poradie[1][i] ^ poradie[1][i + 1]
                poradie[0][i] = poradie[0][i] ^ poradie[0][i + 1]
                poradie[1][i] = poradie[1][i] ^ poradie[1][i + 1]
                t = 0
    for i in range(0, pivot + 1, 1):
        if poradie[0][i] != i:
            return 0
    for i in range(pivot + 1, d_row + 1, 1):
        if poradie[0][i] > d_row:
            return 0
    N = [[0 for i in range(0, c*k +2, 1)] for j in range(0, c*k + 2, 1)]
    for i in range(0, c*k + 2, 1):
        for j in range(0, c*k + 2, 1):
            if i > d_row and j > d_row:
                mod = c2
            else:
                mod = c
            if i != j:
                N[i][j] = divmod((M[perm[poradie[0][i]]][perm[poradie[0][j]]] - M[perm[poradie[0][0]]][perm[poradie[0][j]]] - M[perm[poradie[0][i]]][perm[poradie[0][0]]])*g, mod)[1]
    return N


def ekvivalencia(c, c2, k, p, nu, d_row, matica):
    per = []
    k2 = divmod(c*k, c2)[0]
    m = denum_matrix(c, c2, k, p, nu, c*k + 2, d_row)
    dop = [i for i in range(d_row + 1, c*k + 2, 1)]
    nu2 = num_matrix(c, k, m, d_row)
    if d_row == 2:
        for i in itertools.permutations([0, 1, 2]):
            per.append([1, list(i) + dop])
    else:
        P = open("ZP/P_" + str(c2) + "," + str(k2) + "/P_na_M" + str(matica) + "riadok" + str(d_row) + ".txt", "r")
        for i, s in enumerate(P):
            pe = denum(c*k + 2, d_row + 2, int(s))
            per.append([pe[0], pe[1:] + dop])
        mat = denum_matrix(c, c2, k, p, nu, c*k + 1, d_row)
    for g, perm in per:
        for q in G[c]:
            t = 1
            for i in perm[:d_row + 1]:
                if i > d_row:
                    t = 0
                    print("#WTF", perm)
                    break
            if t == 0:
                continue
            mat = usp_nor_tvar(m, c, c2, k, p, perm, q, d_row, d_row)
            if mat == 0:
                continue
            o = num_matrix(c, k, mat, d_row)
            if o < nu2:
                print(o)
                return 0
    return 1


def perm_minor(M, c, k, p, per, d_row):
    N = copy.deepcopy(M)
    for i in range(d_row, c*k + 2, 1):
        for j in range(d_row, c*k + 2, 1):
            N[i][j] = M[per[i]][per[j]]
    return N


def engine(c2, k2, c3, k3, c, k, p, M, d_row, stop_row, n):
    if d_row == stop_row:
        N = copy.deepcopy(M)
        t = check(c, k, p, N, d_row)
        if t == 1:
            n.write(str(num_matrix(c, k, N, d_row)) + "\n")
            for i in N:
                print(i)
            print("\n")
    else:
        D3 = list()
        perm = [i for i in range(0, c * k + 2, 1)]
        for w in range(d_row, c * k + 2, 1):
            per = [i for i in range(0, c * k + 2, 1)]
            perr = [i for i in range(0, c * k + 2, 1)]
            if d_row == 1:
                for i in range(0, c * k, 1):
                    perr[divmod(i, k)[1] + divmod(k2*divmod(i, k)[0], c*k)[1] + k*divmod(i, k*c2)[0] + 2 ] = i + 2
                for i, j in enumerate(perr):
                    per[j] = i
            N = perm_minor(M, c, k, p, per, d_row)
            if [divmod(N[d_row][j], c2)[1] for j in range(0, d_row, 1)] == [divmod(N[w][j], c2)[1] for j in range(0, d_row, 1)]:
                a = copy.deepcopy(per[d_row])
                b = copy.deepcopy(per[w])
                per[d_row] = b
                per[w] = a
                N = perm_minor(N, c, k, p, per, d_row)
                (nadb2, nadb6, poz) = bloks(N, c2, k2, c3, k3, p, d_row)
                D2 = copy.deepcopy(N[d_row])
                t = 0
                if d_row != 1:
                    for j in nadb2[1:-1]:
                        x = [[D2[l], l] for l in j]
                        x = sorted(x)
                        ll = [per[l[1]] for l in x]
                        for i, l in enumerate(ll):
                            per[j[i]] = l
                W = perm_minor(M, c, k, p, per, d_row)
                if W in D3:
                    continue
                D3.append(copy.deepcopy(W))
                D2 = copy.deepcopy(W[d_row])
                (nadb2, nadb6, poz) = bloks(W, c2, k2, c3, k3, p, d_row)
                D = doplnok(W, c2, k2, c3, k3, p, d_row)
                per = [i for i in range(0, c * k + 2, 1)]
                while per != 0:
                    d = copy.deepcopy(D)
                    N = perm_minor(W, c, k, p, per, d_row)
                    d2 = copy.deepcopy(N[d_row])
                    bloky = bl(c2, nadb6, d2, d_row)
                    poz = [[] for i in range(0, c2, 1)]
                    for i, j in enumerate(d2):
                        if i > d_row:
                            poz[j].append(i)
                    for i, j in enumerate(poz):
                        for l, ll in enumerate(j):
                            N[d_row][ll] = divmod(i + d[i][l] * c2, c)[1]
                    while 0 not in d:
                        t = check(c, k, p, N, d_row)
                        if t == 1:
                            O = usp_nor_tvar(N, c, c2, k, p, perm, 1, d_row, d_row)
                            s = str(num_matrix(c, k, O, c*k + 1))
                            n.write(s + "\n")
                            n.flush()
                        if d[-1] == None:
                            break
                        for i, ii in reversed(list(enumerate(d))):
                            if ii == None:
                                break
                            if ii == []:
                                continue
                            if type(ii) is int:
                                ii == [ii]
                            dd = lex_post_bloky(bloky[i], ii)
                            d[i] = dd
                            if dd != 0:
                                for l, ll in enumerate(d[i + 1:]):
                                    d[i + 1 + l] = copy.deepcopy(D[i + 1 + l])
                                d[i] = dd
                                for j, jj in enumerate(poz[i:]):
                                    for l, ll in enumerate(jj):
                                        N[d_row][ll] = divmod(i + j + d[i + j][l] * c2, c)[1]
                                break
                    per = lex_post_nadbloky(c2, nadb2, nadb6, D2, per)
            if d_row == 1:
                break


def tvar(mat, c, k, p, perm, g):
    M = copy.deepcopy(mat)
    nula = perm[0]
    for i in range(0, c*k + 2, 1):
        for j in range(0, c*k + 2, 1):
            if i != j:
                M[i][j] = divmod((mat[perm[i]][perm[j]] - mat[nula][perm[j]] - mat[perm[i]][nula])*g, c)[1]
    return M


def init(c2, k2, c3, k3, c, k, stop_row, step, matica):
    d = 2
    d_row = 2
    if divmod(k, 2)[1] == 0:
        p = 0
    else:
        p = int(divmod(c * (c - 1) / 2, c)[1])
    M = new_matrix(c, k, p)
    i = step[1]
    if step[0] == 0:
        n = open("N/N_" + str(c) + "," + str(k) + "/N_" + str(c) + "," + str(k) + "_riadok_" + str(i) + "matica_" + str(matica) + ".txt", "w")
        f = open("FN/FN_" + str(c) + "," + str(k) + "/FN_" + str(c) + "," + str(k) + "_riadok_" + str(i - 1) + "matica_" + str(matica) + ".txt", "r")
    else:
        f = open("FN/FN_" + str(c) + "," + str(k) + "/FN_" + str(c) + "," + str(k) + "_riadok_" + str(i) + "matica_" + str(matica) + ".txt", "w")
        n = open("N/N_" + str(c) + "," + str(k) + "/N_" + str(c) + "," + str(k) + "_riadok_" + str(i) + "matica_" + str(matica) + ".txt", "r")
    if i > 2:
        d_row = i
        if step[0] == 0:
            print("#" + str(step))
            for j, s in enumerate(f):
                M = denum_matrix(c, c2, k, p, int(s), c*k + 1, d_row - 1)
                engine(c2, k2, c3, k3, c, k, p, M, d_row, stop_row, n)
                print(j)
        else:
            dup = set()
            print("#" + str(step))
            for j, s in enumerate(n):
                nu = int(s)
                if nu not in dup:
                    dup.add(nu)
                    if ekvivalencia(c, c2, k, p, nu, d_row, matica) == 1:
                        f.write(s)
                        f.flush()
                    print(j)
            dup.clear()
    n.close()
    f.close()


def in_main(c2, k2, c3, k3, stop_row, matica):
    c = c2*c3
    k = divmod(c2*k2, c)[0]
    if divmod(k, 2)[1] == 0:
        p = 0
    else:
        p = int(divmod(c * (c - 1) / 2, c)[1])
    M = new_matrix(c, k, p)
    try:
        os.mkdir("N")
    except OSError:
        pass
    try:
        os.mkdir("FN")
    except OSError:
        pass
    try:
        os.mkdir("N/N_" + str(c) + "," + str(k))
    except OSError:
        pass
    try:
        os.mkdir("FN/FN_" + str(c) + "," + str(k))
    except OSError:
        pass
    d_row = 0
    dup = set()
    f = open("FN/FN_" + str(c) + "," + str(k) + "/FN_" + str(c) + "," + str(k) + "_riadok_" + str(d_row) + "matica_" + str(matica) + ".txt", "w")
    n = open("N/N_" + str(c) + "," + str(k) + "/N_" + str(c) + "," + str(k) + "_riadok_" + str(d_row) + "matica_" + str(matica) + ".txt", "w")
    n.write(str(0) + "\n")
    f.write(str(0) + "\n")
    n.close()
    f.close()
    d_row = 1
    f = open("FN/FN_" + str(c) + "," + str(k) + "/FN_" + str(c) + "," + str(k) + "_riadok_" + str(d_row) + "matica_" + str(matica) + ".txt", "w")
    n = open("N/N_" + str(c) + "," + str(k) + "/N_" + str(c) + "," + str(k) + "_riadok_" + str(d_row) + "matica_" + str(matica) + ".txt", "w")
    f2 = open("FN/FN_" + str(c2) + "," + str(k2) + "/FN_" + str(c2) + "," + str(k2) + "_riadok_" + str(c2*k2 + 1) + ".txt", "r")
    for i, s in enumerate(f2):
        if i != matica:
            continue
        nu = int(s)
        mat = denum_matrix(c2, c2, k2, p, nu, c * k + 1, d_row - 1)
        for g in G[c2]:
            mat2 = usp_nor_tvar(mat, c2, c2, k2, p, [i for i in range(0, c*k + 2, 1)], g, d_row, d_row )
            engine(c2, k2, c3, k3, c, k, p, mat2, d_row, stop_row, n)
            print("#")
    n.close()
    n = open("N/N_" + str(c) + "," + str(k) + "/N_" + str(c) + "," + str(k) + "_riadok_" + str(d_row) + "matica_" + str(matica) + ".txt", "r")
    for i, s in enumerate(n):
        f.write(s)
    n.close()
    f.close()
    d_row = 2
    f = open("FN/FN_" + str(c) + "," + str(k) + "/FN_" + str(c) + "," + str(k) + "_riadok_" + str(d_row - 1) + "matica_" + str(matica) + ".txt", "r")
    n = open("N/N_" + str(c) + "," + str(k) + "/N_" + str(c) + "," + str(k) + "_riadok_" + str(d_row) + "matica_" + str(matica) + ".txt", "w")
    for i, s in enumerate(f):
        nu = int(s)
        mat = denum_matrix(c, c2, k, p, nu, c*k + 1, d_row - 1)
        engine(c2, k2, c3, k3, c, k, p, mat, d_row, stop_row, n)
    f.close()
    n.close()
    d_row = 2
    print("#")
    n = open("N/N_" + str(c) + "," + str(k) + "/N_" + str(c) + "," + str(k) + "_riadok_" + str(d_row) + "matica_" + str(matica) +  ".txt", "r")
    f = open("FN/FN_" + str(c) + "," + str(k) + "/FN_" + str(c) + "," + str(k) + "_riadok_" + str(d_row) + "matica_" + str(matica) + ".txt", "w")
    for i, s in enumerate(n):
        nu = int(s)
        if nu not in dup:
            dup.add(nu)
            if ekvivalencia(c, c2, k, p, nu, d_row, matica) == 1:
                f.write(s)
                f.flush()
    dup.clear()
    n.close()
    f.close()

print("c = ")
c = int(input())
print("k = ")
k = int(input())
print("podkladové c = ")
c2 = int(input())
print("#matice = ")
matica = int(input())


c3 = divmod(c, c2)[0]
k2 = k*c3
k3 = k*c2

print("c2 = " + str(c2))
print("k2 = " + str(k2))
print("c3 = " + str(c3))
print("k3 = " + str(k3))

in_main(c2, k2, c3, k3, c2*k2 + 1, matica)
for i in range(6, 2*(c*k + 2), 1):
        print("&&", i)
        init(c2, k2, c3, k3, c, k, c*k + 1, [divmod(i, 2)[1], divmod(i, 2)[0]], matica)
