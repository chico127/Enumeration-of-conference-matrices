import sys
import copy
import os.path


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


def new_matrix(c, k, p):
    M = []
    row = [0 for i in range(0, k * c + 2, 1)]
    for i in range(0, k * c + 2, 1):
        M.append(copy.deepcopy(row))
    for i in range(0, k * c, 1):
        M[1][i + 2] = divmod(i, k)[0]
        M[i + 2][1] = int(divmod(divmod(i, k)[0] + p, c)[1])
    return M


def doplnok(M, c, k, p, d_row):
    poct = [0 for i in range(0, c, 1)]
    dop = []
    for i in range(1, d_row, 1):
        poct[M[d_row][i]] = poct[M[d_row][i]] + 1
    for i in range(0, c, 1):
        for j in range(poct[i], k, 1):
            dop.append(i)
    d = copy.deepcopy(dop)
    return d


def bloks(M, c, k, p ,d_row):
    r = [0 for i in range(0, c*k + 2, 1)]
    for i in range(1, d_row):
        for j in range(d_row + 1, c*k + 2, 1):
            r[j] = r[j]*c + M[i][j]
    blk = [0, d_row + 1]
    for i in range(d_row + 1, c*k + 1, 1):
        if r[i] != r[i + 1]:
            blk.append(i + 1)
    blk.append(c*k + 2)
    return blk


def lex_post_bloky(bloky, doplnok, ttt):
    b = bloky
    d = doplnok
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
            return d[0:pivot] + d1 + d2



def lex_post_pref(bloky, doplnok, ttt):
    print(bloky, doplnok)
    b = bloky[:3] + [bloky[-1]]
    d = doplnok
    d2 = sorted(d[b[2]:])
    d = d[:b[2]] + d2
    t = 0
    tt = 0
    for i in range(b[1], b[2], 1):
        t = t + d[i]
        if d[i] != 0:
            tt = tt + 1
    while t > ttt*tt:
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
                    if d[i] != 0:
                        tt = tt + 1
                if t <= ttt*tt:
                    print(d)
                    return d
                break


def lex_post_bloky_s(bloky, doplnok, ttt):
    b = bloky
    d = doplnok
    q = 0
    for i in range(len(b)-3, -1, -1): # lebo s prvým blokom sa to hrať nemá, ten je daný
        if i == 0:
            return 0
        pivot = 0
        if i < ttt:
            print("#", d)
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
                if d[i] != 0:
                    tt = tt + 1
            if t <= ttt*tt:
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


def symetrizuj(M, c, k, p, d_row):
    for j in range(1, d_row + 1, 1):
        for i in range(0, c*k + 2, 1):
            if j < i:
                M[i][j] = divmod(M[j][i] + p, c)[1]
    return M


def check(c, k, p, M, dop, d_row):
    for i in range(0, d_row, 1):
        t = 1
        rozdiel = [0 for i in range(0, c, 1)]
        for j in range(0, c*k + 2, 1):
            if i != j and d_row != j:
                rozdiel[divmod(M[i][j]-dop[j], c)[1]] = rozdiel[divmod(M[i][j]-dop[j], c)[1]] + 1
        for j in range(0, c, 1):
            if rozdiel[j] != k:
                t = t * 0
        if t != 1:
            break
    return t


def num(c, arr):
    n = 0
    for i in arr:
        n = n*c + i
    return n


def num_matrix(c, k, M, d_row):
    a = 0
    for j in range(1, d_row + 1, 1):
        for i in range(2, c*k + 2, 1):
            if j < i:
                a = a * c + M[j][i]
    return a


def denum(c, k, num):
    arr = [0 for i in range(0, c*k + 2, 1)]
    for i in range(c*k + 1, -1, -1):
        num, arr[i] = divmod(num, c)
    return arr


def denum_matrix(c, k, p, num, d_row):
    m = [0 for i in range(0, c*k + 2, 1)]
    M = [copy.deepcopy(m) for i in range(0, c*k + 2, 1)]
    for j in range(d_row, 0, -1):
        for i in range(c * k + 1, 1, -1):
            if j < i:
                num, M[j][i] = divmod(num, c)
    return symetrizuj(M, c, k, p, d_row)


def usp_nor_tvar(M, c, k, p, perm, g, d_row, pivot):
    poradie = [[i for i in range(0, c*k + 2, 1)], [0 for i in range(0, c*k + 2, 1)]]
    nula = perm[0]
    for i in range(1, c*k + 2, 1):
        for j in range(0, d_row + 1, 1):
            if j < i:
                poradie[1][i] = poradie[1][i] * c + divmod((M[perm[j]][perm[i]] - M[nula][perm[i]] - M[perm[j]][nula])*g, c)[1]
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
            if i != j:
                N[i][j] = divmod((M[perm[poradie[0][i]]][perm[poradie[0][j]]] - M[perm[poradie[0][0]]][perm[poradie[0][j]]] - M[perm[poradie[0][i]]][perm[poradie[0][0]]])*g, c)[1]
    return N


def ekvivalencia_s(c, k, p, nu, d_row):
    pivot = 2
    mat = denum_matrix(c, k, p, nu, d_row)
    N = list()
    for i in range(0, d_row + 1, 1):
        N.append(int(num_matrix(c, k, mat, i)))
    N_per = list()
    for i in range(0, d_row + 1, 1):
        N_per.append(list())
    bloky = [i for i in range(0, pivot + 2, 1)]
    if d_row > pivot:
        bloky.append(d_row + 1)
    per = [i for i in range(0, d_row + 1, 1)]
    dop = [i for i in range(d_row + 1, c*k + 2, 1)]
    while per != 0:
        for g in G[c]:
            perm = per + dop
            O = usp_nor_tvar(mat, c, k, p, perm, g, d_row, pivot)
            if O == 0:
                continue
            o = num_matrix(c, k, O, pivot)
            if o < N[pivot]:
                return 0
            if o == N[pivot]:
                N_per[pivot].append([g, per[:pivot + 1]])
        per = lex_post_komb(bloky, per)
    t = ekvivalencia(c, k, p, nu, d_row, pivot + 1, N, N_per)
    return t


def ekvivalencia(c, k, p, nu, d_row, pivot, N, N_per):
    mat = denum_matrix(c, k, p, nu, d_row)
    bloky = [i for i in range(0, pivot + 2, 1)]
    if d_row < pivot:
        return 1
    if d_row > pivot:
        bloky.append(d_row + 1)
    for i, [g, per] in enumerate(N_per[pivot - 1]):
        dop = list()
        for j in range(0, c*k + 2, 1):
            if j not in per:
                dop.append(j)
        for j in dop:
            if j < d_row + 1:
                dopl = list()
                for l in dop:
                    if l != j:
                        dopl.append(l)
                perm = per + [j] + dopl
                O = usp_nor_tvar(mat, c, k, p, perm, g, d_row, pivot)
                if O == 0:
                    continue
                o = num_matrix(c, k, O, pivot)
                if o < N[pivot]:
                    return 0
                if o == N[pivot]:
                    N_per[pivot].append([g, per + [j]])
    t = ekvivalencia(c, k, p, nu, d_row, pivot + 1, N, N_per)
    return t


def engine(c, k, p, M, d_row, stop_row, n):
    if d_row == stop_row:
        N = copy.deepcopy(M)
        d = N[d_row]
        N = symetrizuj(N, c, k, p, d_row)
        t = check(c, k, p, N, d, d_row)
        if t == 1:
            N[d_row] = copy.deepcopy(d)
            N = symetrizuj(N, c, k, p, d_row + 1)
            n.write(str(num_matrix(c, k, N, d_row)) + "\n")
    else:
        d = M[d_row][:d_row + 1] + doplnok(M, c, k, p, d_row)
        bloky = bloks(M, c, k, p, d_row)
        N = copy.deepcopy(M)
        N = symetrizuj(N, c, k, p, d_row)
        if M[d_row - 1][d_row + 1] != 0:
            ttt = c
        else:
            ttt = divmod(c + 1, 2)[0]
        while d != 0:
            t = check(c, k, p, N, d, d_row)
            if t == 1:
                N[d_row] = copy.deepcopy(d)
                N = symetrizuj(N, c, k, p, d_row + 1)
                s = str(num_matrix(c, k, N, d_row))
                n.write(s + "\n")
                n.flush()
            d = lex_post_bloky(bloky, d, ttt)


def engine_s(c, k, p, M, d_row, stop_row, n):
    d = M[d_row][:d_row + 1] + doplnok(M, c, k, p, d_row)
    bloky = bloks(M, c, k, p, d_row)
    N = copy.deepcopy(M)
    N = symetrizuj(N, c, k, p, d_row)
    if M[d_row - 1][d_row + 1] != 0:
        ttt = c
    else:
        ttt = divmod(c + 1, 2)[0]
    while d != 0:
        t = check(c, k, p, N, d, d_row)
        if t == 1:
            N[d_row] = copy.deepcopy(d)
            N = symetrizuj(N, c, k, p, d_row + 1)
            s = str(num_matrix(c, k, N, d_row))
            n.write(s + "\n")
            n.flush()
        d = lex_post_bloky_s(bloky, d, ttt)



def init(c, k, stop_row, step):
    d = 2
    d_row = 2
    if divmod(k, 2)[1] == 0:
        p = 0
    else:
        p = int(divmod(c * (c - 1) / 2, c)[1])
    M = new_matrix(c, k, p)
    i = step[1]
    if step[0] == 0:
        n = open("N/N_" + str(c) + "," + str(k) + "/N_" + str(c) + "," + str(k) + "_riadok_" + str(i) + ".txt", "w")
        f = open("FN/FN_" + str(c) + "," + str(k) + "/FN_" + str(c) + "," + str(k) + "_riadok_" + str(i - 1) + ".txt", "r")
    else:
        f = open("FN/FN_" + str(c) + "," + str(k) + "/FN_" + str(c) + "," + str(k) + "_riadok_" + str(i) + ".txt", "w")
        n = open("N/N_" + str(c) + "," + str(k) + "/N_" + str(c) + "," + str(k) + "_riadok_" + str(i) + ".txt", "r")
    if i > 2:
        d_row = i
        if step[0] == 0:
            print("#" + str(step))
            for j, s in enumerate(f):
                M = denum_matrix(c, k, p, int(s), d_row - 1)
                engine(c, k, p, M, d_row, stop_row, n)
                print(j)
        else:
            print("#" + str(step))
            for j, s in enumerate(n):
                nu = int(s)
                matice = 0
                if ekvivalencia_s(c, k, p, nu, d_row) == 1:
                    f.write(s)
                    f.flush()
                print(j)
    n.close()
    f.close()


def in_main(c, k, stop_row):
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
    f = open("FN/FN_" + str(c) + "," + str(k) + "/FN_" + str(c) + "," + str(k) + "_riadok_" + str(d_row) + ".txt", "w")
    n = open("N/N_" + str(c) + "," + str(k) + "/N_" + str(c) + "," + str(k) + "_riadok_" + str(d_row) + ".txt", "w")
    n.write(str(0) + "\n")
    f.write(str(0) + "\n")
    n.close()
    f.close()
    d_row = 1
    f = open("FN/FN_" + str(c) + "," + str(k) + "/FN_" + str(c) + "," + str(k) + "_riadok_" + str(d_row) + ".txt", "w")
    n = open("N/N_" + str(c) + "," + str(k) + "/N_" + str(c) + "," + str(k) + "_riadok_" + str(d_row) + ".txt", "w")
    n.write(str(num_matrix(c, k, M, d_row)) + "\n")
    f.write(str(num_matrix(c, k, M, d_row)) + "\n")
    n.close()
    f.close()
    d_row = 2
    f = open("FN/FN_" + str(c) + "," + str(k) + "/FN_" + str(c) + "," + str(k) + "_riadok_" + str(d_row) + ".txt", "w")
    n = open("N/N_" + str(c) + "," + str(k) + "/N_" + str(c) + "," + str(k) + "_riadok_" + str(d_row) + ".txt", "w")
    engine_s(c, k, p, M, d_row, stop_row, n)
    n.close()
    n = open("N/N_" + str(c) + "," + str(k) + "/N_" + str(c) + "," + str(k) + "_riadok_" + str(d_row) + ".txt", "r")
    for i, s in enumerate(n):
        nu = int(s)
        if ekvivalencia_s(c, k, p, nu, d_row) == 1:
            f.write(s)
    n.close()
    f.close()


print("c =")
c = int(input())
print("min k =")
k1 = int(input())
print("max included k =")
k2 = int(input()) + 1
for k in range(k1, k2, 1):
    in_main(c, k, c*k + 1)
    for i in range(6, 2*c*k + 4, 1):
        print("&&", c, k)
        init(c, k, c*k + 1, [divmod(i, 2)[1], divmod(i, 2)[0]])
