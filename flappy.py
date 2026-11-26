import random
import pygame

pygame.init()

# ---------- Configurações do jogo ----------
LARGURA, ALTURA = 400, 600
CHAO = 520          # y onde o chão começa
FPS = 60

TELA = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("Flappy Bird")
RELOGIO = pygame.time.Clock()
FONTE = pygame.font.SysFont("arial", 36, bold=True)


# ---------- Passaro ----------
class Passaro:
    RAIO = 15
    GRAVIDADE = 0.5   # quanto a velocidade aumenta por frame (para baixo)
    PULO = -8         # velocidade dada ao pular (negativo = para cima)

    def __init__(self):
        self.x = 100
        self.y = 250
        self.vel = 0

    def pular(self):
        self.vel = self.PULO

    def mover(self):
        self.vel += self.GRAVIDADE
        self.y += self.vel

    def desenhar(self, tela):
        pygame.draw.circle(tela, (255, 220, 0), (self.x, int(self.y)), self.RAIO)

    def retangulo(self):
        return pygame.Rect(
            self.x - self.RAIO, int(self.y) - self.RAIO,
            self.RAIO * 2, self.RAIO * 2,
        )


# ---------- Cano (par topo + base com uma abertura no meio) ----------
class Cano:
    LARGURA = 60
    ABERTURA = 150
    VELOCIDADE = 3

    def __init__(self, x):
        self.x = x
        self.altura_topo = random.randint(80, CHAO - self.ABERTURA - 80)
        self.passou = False

    @property
    def y_base(self):
        return self.altura_topo + self.ABERTURA

    def mover(self):
        self.x -= self.VELOCIDADE

    def desenhar(self, tela):
        cor = (40, 180, 60)
        pygame.draw.rect(tela, cor, (self.x, 0, self.LARGURA, self.altura_topo))
        pygame.draw.rect(tela, cor, (self.x, self.y_base, self.LARGURA, CHAO - self.y_base))

    def colide(self, passaro):
        r = passaro.retangulo()
        topo = pygame.Rect(self.x, 0, self.LARGURA, self.altura_topo)
        base = pygame.Rect(self.x, self.y_base, self.LARGURA, CHAO - self.y_base)
        return r.colliderect(topo) or r.colliderect(base)


# ---------- Desenho da tela ----------
def desenhar(passaro, canos, pontos):
    TELA.fill((135, 206, 235))                                  # céu
    for cano in canos:
        cano.desenhar(TELA)
    pygame.draw.rect(TELA, (222, 184, 135), (0, CHAO, LARGURA, ALTURA - CHAO))  # chão
    passaro.desenhar(TELA)
    texto = FONTE.render(str(pontos), True, (255, 255, 255))
    TELA.blit(texto, (LARGURA // 2 - texto.get_width() // 2, 30))
    pygame.display.update()


# ---------- Uma partida ----------
def jogar():
    """Roda uma partida. Retorna True se o passaro morreu (reiniciar)
    e False se o jogador fechou a janela."""
    passaro = Passaro()
    canos = [Cano(500)]
    pontos = 0

    while True:
        RELOGIO.tick(FPS)

        # 1) Entradas do jogador
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                return False
            if evento.type == pygame.KEYDOWN and evento.key == pygame.K_SPACE:
                passaro.pular()
            if evento.type == pygame.MOUSEBUTTONDOWN:
                passaro.pular()

        # 2) Atualiza o mundo
        passaro.mover()
        for cano in canos:
            cano.mover()
            if cano.colide(passaro):
                return True
            if not cano.passou and cano.x + Cano.LARGURA < passaro.x:
                cano.passou = True
                pontos += 1

        # novo cano quando o último já andou o suficiente
        if canos[-1].x < LARGURA - 200:
            canos.append(Cano(LARGURA))
        # remove canos que saíram da tela
        canos = [c for c in canos if c.x + Cano.LARGURA > 0]

        # bateu no chão ou no teto
        if passaro.y + Passaro.RAIO >= CHAO or passaro.y - Passaro.RAIO <= 0:
            return True

        # 3) Desenha
        desenhar(passaro, canos, pontos)


def main():
    while jogar():
        pass  # morreu: começa outra partida
    pygame.quit()


if __name__ == "__main__":
    main()
