"""
Flappy Bird com Q-Learning tabular.

Usa a MESMA fisica/constantes do flappy.py. O ambiente aqui e puro Python
(sem pygame), entao o treino roda rapido. O pygame so e usado no modo 'demo'.

Uso:
  python flappy_q.py train   # treina, salva q_table.pkl, treino.csv, grafico_treino.png
  python flappy_q.py eval    # avalia a politica gulosa em N sementes (resultados.csv)
  python flappy_q.py demo    # mostra o agente treinado jogando (precisa de pygame)
"""
import csv
import pickle
import random
import statistics
import sys
from collections import defaultdict

# ---------------- Constantes (identicas ao flappy.py) ----------------
LARGURA, ALTURA, CHAO = 400, 600, 520
RAIO, GRAVIDADE, PULO, X_PASSARO, Y_INICIAL = 15, 0.5, -8, 100, 250
C_LARGURA, C_ABERTURA, C_VEL = 60, 150, 3

# ---------------- Hiperparametros ----------------
ALFA = 0.1             # taxa de aprendizado
GAMA = 0.99            # fator de desconto
EPS_INI, EPS_FIM = 0.1, 0.002
EPS_DECAI_ATE = 20000   # episodios ate epsilon chegar em EPS_FIM (decai linear)
N_EPISODIOS = 40000
MAX_CANOS_TREINO = 200 # corta o episodio (evita episodios infinitos)
R_VIVO, R_CANO, R_MORTE = 0.1, 10.0, -100.0
BIN_DX, BIN_DY = 20, 10
P_PULO_EXPLORA = 0.15  # na exploracao, pular e mais raro que nao pular
SEMENTE_TREINO = 42
A_CADA, N_VAL = 1000, 30   # validacao: a cada 1000 episodios, 30 sementes (5000..5029)
ALFA_FIM = 0.02            # alfa decai linearmente de ALFA ate ALFA_FIM


# ---------------- Ambiente ----------------
class Cano:
    def __init__(self, x, rng):
        self.x = x
        self.altura_topo = rng.randint(80, CHAO - C_ABERTURA - 80)
        self.y_base = self.altura_topo + C_ABERTURA
        self.passou = False


class FlappyEnv:
    """Estado: (dx, dy, vel) discretizados. Acoes: 0 = nada, 1 = pular."""

    def __init__(self, semente=None):
        self.rng = random.Random(semente)
        self.reset()

    def reset(self):
        self.y, self.vel, self.pontos = float(Y_INICIAL), 0.0, 0
        self.canos = [Cano(500, self.rng)]
        return self.estado()

    def _proximo_cano(self):
        for c in self.canos:
            if c.x + C_LARGURA > X_PASSARO:
                return c
        return self.canos[0]

    def estado(self):
        c = self._proximo_cano()
        dx = max(-3, min(15, int((c.x - X_PASSARO) // BIN_DX)))
        dy = max(-20, min(20, int((c.y_base - C_ABERTURA / 2 - self.y) // BIN_DY)))
        return (dx, dy, max(-8, min(6, int(self.vel))))

    def _colide(self):
        yi = int(self.y)
        if self.y + RAIO >= CHAO or self.y - RAIO <= 0:
            return True
        esq, dir_ = X_PASSARO - RAIO, X_PASSARO + RAIO
        topo, base = yi - RAIO, yi + RAIO
        for c in self.canos:
            if esq < c.x + C_LARGURA and dir_ > c.x:
                if topo < c.altura_topo or base > c.y_base:
                    return True
        return False

    def passo(self, acao):
        """Retorna (novo_estado, recompensa, terminou)."""
        if acao == 1:
            self.vel = PULO
        self.vel += GRAVIDADE
        self.y += self.vel
        recompensa = R_VIVO
        for c in self.canos:
            c.x -= C_VEL
            if not c.passou and c.x + C_LARGURA < X_PASSARO:
                c.passou = True
                self.pontos += 1
                recompensa += R_CANO
        if self.canos[-1].x < LARGURA - 200:
            self.canos.append(Cano(LARGURA, self.rng))
        self.canos = [c for c in self.canos if c.x + C_LARGURA > 0]
        if self._colide():
            return self.estado(), R_MORTE, True
        return self.estado(), recompensa, False


# ---------------- Agente ----------------
def nova_tabela():
    return defaultdict(lambda: [0.0, 0.0])


def melhor_acao(Q, s):
    q = Q[s]
    return 1 if q[1] > q[0] else 0  # empate -> nao pular


def jogar_guloso(Q, semente, meta=100):
    env = FlappyEnv(semente=semente)
    s, fim = env.reset(), False
    while not fim and env.pontos < meta:
        s, _, fim = env.passo(melhor_acao(Q, s))
    return env.pontos


def treinar():
    rng = random.Random(SEMENTE_TREINO)
    Q = nova_tabela()
    historico = []
    melhor_val, melhor_Q = -1.0, None
    for ep in range(1, N_EPISODIOS + 1):
        eps = max(EPS_FIM, EPS_INI - (EPS_INI - EPS_FIM) * ep / EPS_DECAI_ATE)
        alfa = ALFA - (ALFA - ALFA_FIM) * ep / N_EPISODIOS
        env = FlappyEnv(semente=rng.randrange(10**9))
        s, fim = env.reset(), False
        while not fim and env.pontos < MAX_CANOS_TREINO:
            a = (1 if rng.random() < P_PULO_EXPLORA else 0) if rng.random() < eps else melhor_acao(Q, s)
            s2, r, fim = env.passo(a)
            alvo = r if fim else r + GAMA * max(Q[s2])
            Q[s][a] += alfa * (alvo - Q[s][a])
            s = s2
        historico.append((ep, env.pontos, eps, len(Q)))
        if ep % A_CADA == 0:
            val = statistics.mean(jogar_guloso(Q, 5000 + i) for i in range(N_VAL))
            if val > melhor_val:
                melhor_val, melhor_Q = val, {k: list(v) for k, v in Q.items()}
            print(f"   validacao (guloso): {val:.1f} canos | melhor ate agora: {melhor_val:.1f}")
        if ep % 5000 == 0:
            ult = [h[1] for h in historico[-500:]]
            print(f"ep {ep:5d} | media(500)={statistics.mean(ult):7.2f} "
                  f"| max={max(ult):4d} | eps={eps:.3f} | estados={len(Q)}")
    with open("q_table.pkl", "wb") as f:
        pickle.dump(melhor_Q, f)   # salva o melhor checkpoint (pela validacao)
    with open("treino.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episodio", "canos", "epsilon", "estados_visitados"])
        w.writerows(historico)
    grafico(historico)
    return Q


def grafico(historico):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib ausente: grafico nao gerado.")
        return
    eps = [h[0] for h in historico]
    pts = [h[1] for h in historico]
    janela = 100
    media = [statistics.mean(pts[max(0, i - janela + 1):i + 1]) for i in range(len(pts))]
    plt.figure(figsize=(7, 4))
    plt.plot(eps, pts, alpha=0.25, label="Canos por episodio")
    plt.plot(eps, media, linewidth=2, label=f"Media movel ({janela})")
    plt.xlabel("Episodio")
    plt.ylabel("Canos ultrapassados")
    plt.title("Q-Learning no Flappy Bird")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("grafico_treino.png", dpi=200, bbox_inches="tight")
    plt.close()


def carregar_q():
    with open("q_table.pkl", "rb") as f:
        d = pickle.load(f)
    Q = nova_tabela()
    Q.update(d)
    return Q


def avaliar(n=100, meta=100):
    """Politica gulosa em n sementes distintas (diferentes das de treino)."""
    Q = carregar_q()
    resultados = []
    for i in range(n):
        env = FlappyEnv(semente=10_000 + i)
        s, fim = env.reset(), False
        while not fim and env.pontos < meta:
            s, _, fim = env.passo(melhor_acao(Q, s))
        resultados.append(env.pontos)
    sucesso = sum(p >= meta for p in resultados) / n
    print(f"Avaliacao em {n} sementes (meta = {meta} canos)")
    print(f"  media = {statistics.mean(resultados):.1f} | mediana = {statistics.median(resultados):.1f} "
          f"| min = {min(resultados)} | max = {max(resultados)} | taxa de sucesso = {sucesso:.0%}")
    with open("resultados.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["semente", "canos"])
        for i, p in enumerate(resultados):
            w.writerow([10_000 + i, p])


# ---------------- Demonstracao visual ----------------
def demo():
    import pygame
    pygame.init()
    tela = pygame.display.set_mode((LARGURA, ALTURA))
    pygame.display.set_caption("Flappy Bird - Q-Learning")
    relogio = pygame.time.Clock()
    fonte = pygame.font.SysFont("arial", 36, bold=True)
    Q = carregar_q()
    while True:
        env = FlappyEnv()
        s, fim = env.reset(), False
        while not fim:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    return
            s, _, fim = env.passo(melhor_acao(Q, s))
            tela.fill((135, 206, 235))
            for c in env.canos:
                pygame.draw.rect(tela, (40, 180, 60), (c.x, 0, C_LARGURA, c.altura_topo))
                pygame.draw.rect(tela, (40, 180, 60), (c.x, c.y_base, C_LARGURA, CHAO - c.y_base))
            pygame.draw.rect(tela, (222, 184, 135), (0, CHAO, LARGURA, ALTURA - CHAO))
            pygame.draw.circle(tela, (255, 220, 0), (X_PASSARO, int(env.y)), RAIO)
            txt = fonte.render(str(env.pontos), True, (255, 255, 255))
            tela.blit(txt, (LARGURA // 2 - txt.get_width() // 2, 30))
            pygame.display.update()
            relogio.tick(60)


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "train"
    if modo == "train":
        treinar()
        avaliar()
    elif modo == "eval":
        avaliar()
    elif modo == "demo":
        demo()
    else:
        print(__doc__)
