"""
Flappy Bird com Q-Learning tabular.

Usa a MESMA fisica/constantes do flappy.py. O ambiente aqui e puro Python
(sem pygame), entao o treino roda rapido. O pygame so e usado no modo 'demo'.

Uso:
  python flappy_q.py train   # treina, salva q_table.pkl, treino.csv, grafico_treino.png
  python flappy_q.py eval    # avalia a politica gulosa em N sementes (resultados.csv)
  python flappy_q.py train_visual  # treina COM janela (tecla F = modo rapido)
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


# ---------------- Janela visual (pygame, so importado se for usado) ----------------
class Janela:
    """Desenha o jogo. Tecla F alterna entre modo normal (60 FPS) e rapido."""

    def __init__(self, titulo):
        import pygame
        self.pg = pygame
        pygame.init()
        self.tela = pygame.display.set_mode((LARGURA, ALTURA))
        pygame.display.set_caption(titulo)
        self.relogio = pygame.time.Clock()
        self.fonte = pygame.font.SysFont("arial", 36, bold=True)
        self.pequena = pygame.font.SysFont("arial", 18, bold=True)
        self.rapido = False
        self.passos = 0

    def quadro(self, env, linhas=()):
        pg = self.pg
        for e in pg.event.get():
            if e.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if e.type == pg.KEYDOWN and e.key == pg.K_f:
                self.rapido = not self.rapido
        self.passos += 1
        if self.rapido and self.passos % 30:   # modo rapido: desenha 1 a cada 30 passos
            return
        self.tela.fill((135, 206, 235))
        for c in env.canos:
            pg.draw.rect(self.tela, (40, 180, 60), (c.x, 0, C_LARGURA, c.altura_topo))
            pg.draw.rect(self.tela, (40, 180, 60), (c.x, c.y_base, C_LARGURA, CHAO - c.y_base))
        pg.draw.rect(self.tela, (222, 184, 135), (0, CHAO, LARGURA, ALTURA - CHAO))
        pg.draw.circle(self.tela, (255, 220, 0), (X_PASSARO, int(env.y)), RAIO)
        txt = self.fonte.render(str(env.pontos), True, (255, 255, 255))
        self.tela.blit(txt, (LARGURA // 2 - txt.get_width() // 2, 30))
        for i, linha in enumerate(linhas):
            self.tela.blit(self.pequena.render(linha, True, (255, 255, 255)), (10, 10 + i * 22))
        pg.display.update()
        if not self.rapido:
            self.relogio.tick(60)


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


def treinar(visual=False):
    janela = Janela("Flappy Bird - Treino Q-Learning") if visual else None
    recorde = 0
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
            if janela:
                recorde = max(recorde, env.pontos)
                janela.quadro(env, (f"Episodio: {ep}/{N_EPISODIOS}", f"Epsilon: {eps:.3f}",
                                    f"Recorde: {recorde}", "F = modo rapido"))
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
    janela = Janela("Flappy Bird - Q-Learning")
    Q = carregar_q()
    while True:
        env = FlappyEnv()
        s, fim = env.reset(), False
        while not fim:
            s, _, fim = env.passo(melhor_acao(Q, s))
            janela.quadro(env, ("Agente treinado", "F = modo rapido"))


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "train"
    if modo in ("train", "train_visual"):
        treinar(visual=(modo == "train_visual"))
        avaliar()
    elif modo == "eval":
        avaliar()
    elif modo == "demo":
        demo()
    else:
        print(__doc__)
