#!/usr/bin/env python3
"""
Gerador de painel comparativo de Data Augmentation
Projeto: sistema anti-esmagamento de leitoes em baia de maternidade

Pega UMA imagem real do dataset e gera um painel PNG mostrando o efeito de
cada transformacao considerada no planejamento da Atividade 6.

Se existir um arquivo de rotulo YOLO (.txt) com o mesmo nome da imagem, as
caixas sao desenhadas e transformadas junto com a imagem. Esse e o ponto
conceitual da figura: a imagem muda, a caixa acompanha, e a resposta certa
continua a mesma.

Instalacao
    pip install opencv-python numpy pillow

Uso
    python gerar_painel_augmentation.py caminho/da/imagem.jpg
    python gerar_painel_augmentation.py imagem.jpg --saida painel.png
    python gerar_painel_augmentation.py imagem.jpg --sem-caixas

O script procura o rotulo automaticamente em:
    mesma pasta da imagem, com extensao .txt
    ../labels/<mesmo nome>.txt        (estrutura padrao do Roboflow / YOLO)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Aparencia do painel
# ---------------------------------------------------------------------------

LARGURA_CELULA = 320
ALTURA_CELULA = 240
MARGEM = 18
ALTURA_RODAPE = 52
COLUNAS = 4

COR_FUNDO = (255, 255, 255)
COR_TITULO = (26, 26, 26)
COR_SUBTITULO = (110, 110, 110)
COR_DESCARTE = (163, 45, 45)
COR_CAIXA = (255, 210, 40)      # BGR no OpenCV


def carregar_fonte(tamanho, negrito=False):
    candidatos = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if negrito
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if negrito else "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if negrito
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for c in candidatos:
        try:
            return ImageFont.truetype(c, tamanho)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Leitura de imagem e rotulo
# ---------------------------------------------------------------------------

def ler_imagem(caminho: Path):
    """Leitura segura, funciona com caminhos acentuados no Windows."""
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    if dados.size == 0:
        return None
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def achar_rotulo(img_path: Path):
    """Procura o .txt correspondente nos locais usuais."""
    direto = img_path.with_suffix(".txt")
    if direto.exists():
        return direto
    alternativo = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    if alternativo.exists():
        return alternativo
    return None


def ler_caixas(rotulo: Path, largura: int, altura: int):
    """
    Le rotulos YOLO (classe xc yc w h, normalizados) e devolve caixas em
    pixels no formato (x1, y1, x2, y2).
    """
    caixas = []
    if rotulo is None:
        return caixas
    for linha in rotulo.read_text(encoding="utf-8").strip().splitlines():
        partes = linha.split()
        if len(partes) < 5:
            continue
        _, xc, yc, w, h = partes[0], *map(float, partes[1:5])
        cx, cy, bw, bh = xc * largura, yc * altura, w * largura, h * altura
        caixas.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return caixas


def desenhar_caixas(img, caixas, espessura=2):
    saida = img.copy()
    alt, larg = saida.shape[:2]
    for (x1, y1, x2, y2) in caixas:
        x1 = int(max(0, min(larg - 1, x1)))
        y1 = int(max(0, min(alt - 1, y1)))
        x2 = int(max(0, min(larg - 1, x2)))
        y2 = int(max(0, min(alt - 1, y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cv2.rectangle(saida, (x1, y1), (x2, y2), COR_CAIXA, espessura)
    return saida


# ---------------------------------------------------------------------------
# Transformacoes  (imagem e caixas juntas)
# ---------------------------------------------------------------------------

def t_original(img, caixas):
    return img.copy(), list(caixas)


def t_rotacao(img, caixas, angulo=18):
    alt, larg = img.shape[:2]
    M = cv2.getRotationMatrix2D((larg / 2, alt / 2), angulo, 1.0)
    saida = cv2.warpAffine(img, M, (larg, alt), borderMode=cv2.BORDER_REPLICATE)

    novas = []
    for (x1, y1, x2, y2) in caixas:
        cantos = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        cantos = np.hstack([cantos, np.ones((4, 1), dtype=np.float32)])
        rot = cantos @ M.T
        novas.append((rot[:, 0].min(), rot[:, 1].min(),
                      rot[:, 0].max(), rot[:, 1].max()))
    return saida, novas


def t_espelho(img, caixas):
    alt, larg = img.shape[:2]
    saida = cv2.flip(img, 1)
    novas = [(larg - x2, y1, larg - x1, y2) for (x1, y1, x2, y2) in caixas]
    return saida, novas


def t_brilho(img, caixas, fator=0.32):
    """Escurece simulando a comutacao para o periodo noturno."""
    saida = np.clip(img.astype(np.float32) * fator, 0, 255).astype(np.uint8)
    return saida, list(caixas)


def t_cinza(img, caixas):
    """Aproxima a resposta do sensor sob iluminacao infravermelha."""
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR), list(caixas)


def t_desfoque(img, caixas, proporcao=0.045):
    """Desfoque direcional, simulando a porca em movimento."""
    larg = img.shape[1]
    tamanho = max(9, int(larg * proporcao) | 1)
    kernel = np.zeros((tamanho, tamanho), dtype=np.float32)
    kernel[tamanho // 2, :] = 1.0 / tamanho
    return cv2.filter2D(img, -1, kernel), list(caixas)


def t_oclusao(img, caixas, fracao=0.28):
    """Apagamento aleatorio, simulando leitao coberto pelo corpo da porca."""
    saida = img.copy()
    alt, larg = saida.shape[:2]
    bw, bh = int(larg * fracao), int(alt * fracao)
    x = int(larg * 0.18)
    y = int(alt * 0.52)
    saida[y:y + bh, x:x + bw] = (70, 70, 70)
    return saida, list(caixas)


def t_escala(img, caixas, fator=1.9):
    """DESCARTADA. Amplia a cena e distorce a razao de tamanho porca/leitao."""
    alt, larg = img.shape[:2]
    M = cv2.getRotationMatrix2D((larg / 2, alt / 2), 0, fator)
    saida = cv2.warpAffine(img, M, (larg, alt), borderMode=cv2.BORDER_REPLICATE)

    novas = []
    for (x1, y1, x2, y2) in caixas:
        cantos = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        cantos = np.hstack([cantos, np.ones((4, 1), dtype=np.float32)])
        esc = cantos @ M.T
        novas.append((esc[:, 0].min(), esc[:, 1].min(),
                      esc[:, 0].max(), esc[:, 1].max()))
    return saida, novas


PAINEIS = [
    ("Original", "imagem do dataset", t_original, False),
    ("Rotacao", "cada granja monta a camera diferente", t_rotacao, False),
    ("Espelhamento", "nao ha lado preferencial", t_espelho, False),
    ("Brilho reduzido", "ciclo dia-noite", t_brilho, False),
    ("Escala de cinza", "modo infravermelho noturno", t_cinza, False),
    ("Desfoque de movimento", "porca se move ao deitar", t_desfoque, False),
    ("Oclusao parcial", "leitao coberto pela porca", t_oclusao, False),
    ("Distorcao de escala", "DESCARTADA - ver texto", t_escala, True),
]


# ---------------------------------------------------------------------------
# Montagem do painel
# ---------------------------------------------------------------------------

def ajustar(img):
    """Redimensiona preservando proporcao e completa com branco."""
    alt, larg = img.shape[:2]
    escala = min(LARGURA_CELULA / larg, ALTURA_CELULA / alt)
    nova = cv2.resize(img, (max(1, int(larg * escala)), max(1, int(alt * escala))),
                      interpolation=cv2.INTER_AREA)
    tela = np.full((ALTURA_CELULA, LARGURA_CELULA, 3), 255, dtype=np.uint8)
    y = (ALTURA_CELULA - nova.shape[0]) // 2
    x = (LARGURA_CELULA - nova.shape[1]) // 2
    tela[y:y + nova.shape[0], x:x + nova.shape[1]] = nova
    return tela


def montar(img, caixas, mostrar_caixas: bool):
    linhas = (len(PAINEIS) + COLUNAS - 1) // COLUNAS
    largura = COLUNAS * LARGURA_CELULA + (COLUNAS + 1) * MARGEM
    altura = linhas * (ALTURA_CELULA + ALTURA_RODAPE) + (linhas + 1) * MARGEM + 34

    tela = Image.new("RGB", (largura, altura), COR_FUNDO)
    desenho = ImageDraw.Draw(tela)
    f_cab = carregar_fonte(19, negrito=True)
    f_tit = carregar_fonte(16, negrito=True)
    f_sub = carregar_fonte(13)

    desenho.text((MARGEM, 10), "Data augmentation aplicada a uma imagem real da baia",
                 font=f_cab, fill=COR_TITULO)

    for i, (titulo, subtitulo, funcao, descartada) in enumerate(PAINEIS):
        col, lin = i % COLUNAS, i // COLUNAS
        x = MARGEM + col * (LARGURA_CELULA + MARGEM)
        y = 34 + MARGEM + lin * (ALTURA_CELULA + ALTURA_RODAPE + MARGEM)

        res, novas = funcao(img, caixas)
        if mostrar_caixas and novas:
            res = desenhar_caixas(res, novas)
        celula = ajustar(res)

        tela.paste(Image.fromarray(cv2.cvtColor(celula, cv2.COLOR_BGR2RGB)), (x, y))

        borda = COR_DESCARTE if descartada else (205, 203, 195)
        largura_borda = 3 if descartada else 1
        desenho.rectangle([x, y, x + LARGURA_CELULA - 1, y + ALTURA_CELULA - 1],
                          outline=borda, width=largura_borda)

        cor_t = COR_DESCARTE if descartada else COR_TITULO
        cor_s = COR_DESCARTE if descartada else COR_SUBTITULO
        desenho.text((x, y + ALTURA_CELULA + 9), titulo, font=f_tit, fill=cor_t)
        desenho.text((x, y + ALTURA_CELULA + 30), subtitulo, font=f_sub, fill=cor_s)

    return tela


def main():
    p = argparse.ArgumentParser(
        description="Gera painel comparativo de data augmentation")
    p.add_argument("imagem", help="caminho de uma imagem do dataset")
    p.add_argument("--saida", default="painel_augmentation.png",
                   help="arquivo PNG de saida")
    p.add_argument("--rotulo", default=None,
                   help="caminho do .txt YOLO (detectado automaticamente se omitido)")
    p.add_argument("--sem-caixas", action="store_true",
                   help="nao desenha as caixas mesmo se o rotulo existir")
    args = p.parse_args()

    img_path = Path(args.imagem)
    if not img_path.is_file():
        print("Imagem nao encontrada: %s" % img_path)
        sys.exit(1)

    img = ler_imagem(img_path)
    if img is None:
        print("Nao foi possivel ler a imagem: %s" % img_path)
        sys.exit(1)

    alt, larg = img.shape[:2]
    rotulo = Path(args.rotulo) if args.rotulo else achar_rotulo(img_path)
    caixas = [] if args.sem_caixas else ler_caixas(rotulo, larg, alt)

    if args.sem_caixas:
        print("Caixas desativadas por opcao.")
    elif caixas:
        print("Rotulo encontrado: %s  (%d caixas)" % (rotulo, len(caixas)))
    else:
        print("Nenhum rotulo encontrado. Gerando painel apenas com as imagens.")

    painel = montar(img, caixas, mostrar_caixas=bool(caixas))
    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    painel.save(saida, "PNG")

    print("Imagem de origem: %s  (%dx%d)" % (img_path.name, larg, alt))
    print("Painel salvo em:  %s" % saida.resolve())


if __name__ == "__main__":
    main()