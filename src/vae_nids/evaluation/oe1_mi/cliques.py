"""Agrupación por clique para OE1-IM.

En el EDA original NO hay código de clique: los clusters salen de un
union-find a |r| >= 0.9 (notebooks/eda_cicids2017.ipynb, celda 38) y la
resolución por clique a |r| >= 0.95 se hizo a mano en
notebooks/cluster_review_oe1.md. Este módulo formaliza esa resolución
manual como un algoritmo explícito, para poder aplicarla igual a
cualquier criterio (P, L, A_k, S):

  1. Sobre los nodos aún libres, enumerar las cliques maximales (tamaño >= 2)
     con Bron-Kerbosch con pivote (networkx no es dependencia del proyecto).
  2. Tomar solo las de tamaño máximo s y agruparlas por solapamiento.
  3. En cada componente de solapamiento:
       a. una sola clique -> se fusiona (clusters 0, 1, 5 y los pares);
       b. varias -> buscar los empaquetamientos (subconjuntos de cliques
          disjuntas) de cardinalidad máxima; si es único, se fusionan esas
          cliques (cluster 3: la cadena BwdMean-BwdMin-FwdMin-FwdMean se
          resuelve en los dos pares direccionales, que es el único
          empaquetamiento de 2);
       c. si no es único -> se fusionan la intersección de todas las cliques
          de la componente y la parte privada de cada clique (miembros que
          no están en ninguna otra), cada una solo si tiene >= 2 nodos; los
          nodos disputados (en algunas cliques pero no en todas) quedan
          libres para la siguiente ronda. Cluster 2: dos 5-cliques que
          comparten el núcleo de 4 -> se fusiona el núcleo; las partes
          privadas (Bwd IAT Max, Idle Min) son de 1 nodo y quedan sueltas.
          Con dos 7-cliques que comparten 1 nodo, se fusionan las dos partes
          privadas de 6 y el nodo compartido queda suelto;
       d. si nada tiene >= 2 nodos -> ambigüedad sin resolver: esos nodos se
          retiran sin fusionar y se reporta.
  4. Retirar los nodos fusionados y repetir hasta que no queden aristas.
"""
from itertools import combinations


def bron_kerbosch(adj: dict[str, set[str]], nodes: set[str]) -> list[frozenset[str]]:
    cliques: list[frozenset[str]] = []
    sub = {u: adj[u] & nodes for u in nodes}

    def expand(r: set[str], p: set[str], x: set[str]) -> None:
        if not p and not x:
            if len(r) >= 2:
                cliques.append(frozenset(r))
            return
        pivot = max(p | x, key=lambda u: len(sub[u] & p))
        for v in sorted(p - sub[pivot]):
            expand(r | {v}, p & sub[v], x & sub[v])
            p = p - {v}
            x = x | {v}

    expand(set(), set(nodes), set())
    return cliques


def _overlap_components(cliques: list[frozenset[str]]) -> list[list[frozenset[str]]]:
    comps: list[list[frozenset[str]]] = []
    remaining = list(cliques)
    while remaining:
        comp = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            for c in list(remaining):
                if any(c & d for d in comp):
                    comp.append(c)
                    remaining.remove(c)
                    changed = True
        comps.append(comp)
    return comps


def _max_packings(cliques: list[frozenset[str]], limit: int = 5000) -> tuple[int, list[tuple[int, ...]], bool]:
    """Todos los empaquetamientos (índices de cliques disjuntas) de
    cardinalidad máxima. Búsqueda exacta con poda; `limit` corta la
    enumeración si explota (se reporta como truncada)."""
    n = len(cliques)
    best = [0]
    sols: list[tuple[int, ...]] = []
    truncated = [False]

    def rec(i: int, chosen: list[int], used: frozenset[str]) -> None:
        if truncated[0]:
            return
        if len(chosen) + (n - i) < best[0]:
            return
        if i == n:
            if len(chosen) > best[0]:
                best[0] = len(chosen)
                sols.clear()
            if len(chosen) == best[0]:
                sols.append(tuple(chosen))
                if len(sols) > limit:
                    truncated[0] = True
            return
        if not (cliques[i] & used):
            rec(i + 1, chosen + [i], used | cliques[i])
        rec(i + 1, chosen, used)

    rec(0, [], frozenset())
    return best[0], sols, truncated[0]


def resolve_groups(nodes: list[str], edges: set[frozenset[str]]) -> dict:
    """Aplica la regla de resolución descrita en el docstring del módulo.

    Devuelve {"groups": [sorted list], "ambiguous": [...], "log": [...]}.
    Cada grupo es una lista ordenada de >= 2 features a fusionar en un
    representante."""
    adj: dict[str, set[str]] = {u: set() for u in nodes}
    for e in edges:
        a, b = tuple(e)
        adj[a].add(b)
        adj[b].add(a)

    free = {u for u in nodes if adj[u]}
    groups: list[list[str]] = []
    ambiguous: list[dict] = []
    log: list[str] = []

    while True:
        cliques = bron_kerbosch(adj, free)
        if not cliques:
            break
        s = max(len(c) for c in cliques)
        top = sorted([c for c in cliques if len(c) == s], key=lambda c: sorted(c))
        progressed = False
        for comp in _overlap_components(top):
            if len(comp) == 1:
                groups.append(sorted(comp[0]))
                free -= comp[0]
                log.append(f"s={s}: clique única {sorted(comp[0])}")
                progressed = True
                continue
            k, sols, truncated = _max_packings(comp)
            if len(sols) == 1 and not truncated:
                for idx in sols[0]:
                    groups.append(sorted(comp[idx]))
                    free -= comp[idx]
                log.append(f"s={s}: {len(comp)} cliques solapadas -> empaquetamiento único de {k}")
                progressed = True
                continue
            inter = frozenset.intersection(*comp)
            privates = [c - frozenset.union(*[d for d in comp if d is not c]) for c in comp]
            fused = [p for p in [inter, *privates] if len(p) >= 2]
            if fused:
                for part in fused:
                    groups.append(sorted(part))
                    free -= part
                log.append(f"s={s}: {len(comp)} cliques solapadas, empaquetamiento no único "
                           f"-> se fusionan intersección/partes privadas {[sorted(p) for p in fused]}; "
                           f"nodos disputados quedan libres")
                progressed = True
            else:
                union = frozenset.union(*comp)
                ambiguous.append({"size": s, "cliques": [sorted(c) for c in comp],
                                  "truncated": truncated})
                free -= union
                log.append(f"s={s}: ambigüedad sin resolver en {sorted(union)} (se deja sin fusionar)")
                progressed = True
        if not progressed:
            break
        # Nodos que quedaron sin aristas hacia otros libres ya no participan.
        free = {u for u in free if adj[u] & free}

    groups.sort(key=lambda g: (-len(g), g))
    return {"groups": groups, "ambiguous": ambiguous, "log": log}


def edges_from_matrix(features: list[str], values, threshold: float) -> set[frozenset[str]]:
    """Aristas {a, b} con values[i, j] >= threshold (i < j)."""
    out: set[frozenset[str]] = set()
    for i, j in combinations(range(len(features)), 2):
        v = values[i, j]
        if v == v and v >= threshold:  # NaN-safe
            out.add(frozenset((features[i], features[j])))
    return out


def choose_representative(group: list[str], kept_51: set[str], entropy: dict[str, float],
                          documented_rep: dict[frozenset[str], str] | None = None,
                          u_given: dict[tuple[str, str], float] | None = None) -> str:
    """Representante de un grupo fusionado.

    1. Si el grupo coincide exactamente con un grupo documentado del EDA, su
       representante documentado.
    2. Si no: entre los miembros que ya están en las 51 (o todos, si ninguno
       lo está), el que más información explica de los demás miembros:
       max media_j u(j | rep) = I(j; rep) / H(j) (`u_given[(j, rep)]`). Es el
       criterio teórico-informacional natural para "a quién conservar": el que
       deja menos entropía sin explicar en los descartados. Desempate por
       entropía propia."""
    if documented_rep and frozenset(group) in documented_rep:
        return documented_rep[frozenset(group)]
    pool = [f for f in group if f in kept_51] or list(group)

    def score(f: str) -> tuple:
        others = [g for g in group if g != f]
        mean_u = (sum(u_given.get((g, f), 0.0) for g in others) / len(others)) if u_given else 0.0
        return (round(mean_u, 12), entropy.get(f, 0.0), f)

    return max(pool, key=score)
