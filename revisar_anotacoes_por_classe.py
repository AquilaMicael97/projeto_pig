#!/usr/bin/env python3
"""
Revisao semantica das anotacoes, por classe
Projeto: sistema anti-esmagamento de leitoes em baia de maternidade

Diferente da triagem de qualidade (a foto esta boa?) e da validacao
estrutural (o arquivo .txt esta bem formado?), este script serve pra
responder uma terceira pergunta: a anotacao esta CERTA? Ou seja, uma caixa
marcada como "sow-sit" realmente mostra a porca sentada, e nao em pe ou
deitada?

Isso nao da pra validar por codigo — precisa de olho humano. O que o
script faz e amostrar algumas imagens de cada classe (priorizando as
classes raras, onde um erro de rotulo pesa mais) e montar um painel por
classe, com todas as caixas da imagem desenhadas e coloridas por classe,
pra revisao visual rapida.

Uso
    python revisar_anotacoes_por_classe.py sows_farm_02.v2i.yolo26
    python revisar_anotacoes_por_classe.py sows_farm_02.v2i.yolo26 --por-classe 8
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

CELULA = 260
MARGEM = 14
RODAPE = 20
COLUNAS = 4
EXTENSOES_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "valid", "test")

CORES = [
    (0, 210, 255),   # piglet - amarelo
    (255, 120, 0),   # sow-sit - azul
    (0, 200, 0),     # sow-sleep - verde
    (0, 130, 255),   # sow-sleep-lactate - laranja
    (255, 0, 255),   # sow-stand - magenta
    (150, 0, 150),   # sow-stand-feed - roxo
]


def ler_imagem(caminho: Path):
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    if dados.size == 0:
        return None
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def ler_caixas(rotulo: Path):
    caixas = []
    if not rotulo.is_file():
        return caixas
    for linha in rotulo.read_text(encoding="utf-8").strip().splitlines():
        p = linha.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        xc, yc, w, h = map(float, p[1:5])
        caixas.append((c, xc, yc, w, h))
    return caixas


def desenhar(img, caixas, nomes):
    saida = img.copy()
    alt, larg = saida.shape[:2]
    for (c, xc, yc, w, h) in caixas:
        cx, cy, bw, bh = xc * larg, yc * alt, w * larg, h * alt
        x1, y1 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
        x2, y2 = int(min(larg - 1, cx + bw / 2)), int(min(alt - 1, cy + bh / 2))
        cor = CORES[c % len(CORES)]
        cv2.rectangle(saida, (x1, y1), (x2, y2), cor, 2)
        cv2.putText(saida, nomes[c][:12], (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor, 1, cv2.LINE_AA)
    return saida


def ajustar(img):
    alt, larg = img.shape[:2]
    escala = min(CELULA / larg, CELULA / alt)
    nova = cv2.resize(img, (max(1, int(larg * escala)), max(1, int(alt * escala))))
    tela = np.full((CELULA, CELULA, 3), 255, dtype=np.uint8)
    y, x = (CELULA - nova.shape[0]) // 2, (CELULA - nova.shape[1]) // 2
    tela[y:y + nova.shape[0], x:x + nova.shape[1]] = nova
    return tela


def carregar_fonte(tam):
    for c in ("C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(c, tam)
        except OSError:
            continue
    return ImageFont.load_default()


def montar_painel(amostras, nomes, titulo, saida_path):
    n = len(amostras)
    colunas = min(COLUNAS, n) or 1
    linhas = (n + colunas - 1) // colunas
    largura = colunas * CELULA + (colunas + 1) * MARGEM
    altura = 34 + linhas * (CELULA + RODAPE) + (linhas + 1) * MARGEM

    tela = Image.new("RGB", (largura, altura), (255, 255, 255))
    desenho = ImageDraw.Draw(tela)
    f_tit = carregar_fonte(17)
    f_leg = carregar_fonte(12)
    desenho.text((MARGEM, 8), titulo, font=f_tit, fill=(20, 20, 20))

    for i, (img_path, split, caixas) in enumerate(amostras):
        img = ler_imagem(img_path)
        if img is None:
            continue
        com_caixas = desenhar(img, caixas, nomes)
        celula = ajustar(com_caixas)

        col, lin = i % colunas, i // colunas
        x = MARGEM + col * (CELULA + MARGEM)
        y = 34 + MARGEM + lin * (CELULA + RODAPE + MARGEM)
        tela.paste(Image.fromarray(cv2.cvtColor(celula, cv2.COLOR_BGR2RGB)), (x, y))
        desenho.rectangle([x, y, x + CELULA - 1, y + CELULA - 1], outline=(200, 200, 200))
        desenho.text((x, y + CELULA + 3), "%s | %s" % (split, img_path.name[:34]),
                     font=f_leg, fill=(60, 60, 60))

    tela.save(saida_path)


def main():
    p = argparse.ArgumentParser(description="Amostra imagens por classe para revisao visual da anotacao")
    p.add_argument("raiz", help="pasta do dataset (contem data.yaml)")
    p.add_argument("--por-classe", type=int, default=8, dest="por_classe",
                   help="quantas imagens amostrar por classe (padrao 8)")
    p.add_argument("--seed", type=int, default=42, help="semente da amostragem (padrao 42)")
    p.add_argument("--saida", default="revisao_anotacoes", help="pasta de saida dos paineis")
    args = p.parse_args()

    raiz = Path(args.raiz)
    cfg = yaml.safe_load((raiz / "data.yaml").read_text(encoding="utf-8"))
    nomes = cfg["names"]

    por_classe = defaultdict(list)  # classe -> [(img_path, split, caixas), ...]
    for split in SPLITS:
        pasta_lbl = raiz / split / "labels"
        pasta_img = raiz / split / "images"
        if not pasta_lbl.is_dir():
            continue
        for lbl in pasta_lbl.glob("*.txt"):
            caixas = ler_caixas(lbl)
            if not caixas:
                continue
            img_path = next((pasta_img / (lbl.stem + ext) for ext in EXTENSOES_IMG
                             if (pasta_img / (lbl.stem + ext)).is_file()), None)
            if img_path is None:
                continue
            classes_na_imagem = {c for c, *_ in caixas}
            for c in classes_na_imagem:
                por_classe[c].append((img_path, split, caixas))

    random.seed(args.seed)
    saida_dir = Path(args.saida)
    saida_dir.mkdir(parents=True, exist_ok=True)

    for c, nome in enumerate(nomes):
        candidatos = por_classe.get(c, [])
        if not candidatos:
            print("classe %-20s: nenhuma imagem encontrada" % nome)
            continue
        amostra = random.sample(candidatos, min(args.por_classe, len(candidatos)))
        arq_saida = saida_dir / ("revisao_classe_%d_%s.png" % (c, nome))
        titulo = "%s  (classe %d, %d instancias no dataset, amostra de %d imagens)" % (
            nome, c, len(candidatos), len(amostra))
        montar_painel(amostra, nomes, titulo, arq_saida)
        print("classe %-20s: %3d imagens com essa classe -> painel em %s" %
              (nome, len(candidatos), arq_saida))


if __name__ == "__main__":
    main()
