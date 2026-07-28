#!/usr/bin/env python3
"""
Validacao estrutural dos rotulos YOLO
Projeto: sistema anti-esmagamento de leitoes em baia de maternidade

Diferente da triagem de qualidade (que julga a IMAGEM: nitidez, brilho,
duplicata), este script valida a integridade do ROTULO:

    - toda imagem tem um .txt correspondente, e vice-versa
    - cada linha do .txt tem exatamente 5 campos (class x y w h)
    - class_id esta dentro do intervalo definido no data.yaml (0..nc-1)
    - x, y, w, h sao floats validos, x/y em [0,1], w/h em (0,1] (sem caixa
      degenerada ou fora da imagem)

Erros aqui sao BUGS de dado, nao julgamento de qualidade: o script termina
com codigo de saida != 0 se encontrar qualquer erro estrutural, para poder
ser usado como gate antes de gerar o dataset_v1.0.

Uso
    python validar_estrutura.py sows_farm_02.v2i.yolo26
"""

import argparse
import csv
import sys
from pathlib import Path

import yaml

EXTENSOES_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "valid", "test")


def validar_linha(linha: str, nc: int):
    """Devolve None se a linha estiver ok, ou uma string com o motivo do erro."""
    partes = linha.split()
    if len(partes) != 5:
        return "linha_com_%d_campos" % len(partes)

    try:
        classe = int(partes[0])
    except ValueError:
        return "class_id_nao_numerico"
    if not (0 <= classe < nc):
        return "class_id_fora_do_intervalo(%d)" % classe

    try:
        x, y, w, h = (float(v) for v in partes[1:5])
    except ValueError:
        return "coordenada_nao_numerica"

    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return "centro_fora_de_[0,1]"
    if w <= 0.0 or h <= 0.0:
        return "caixa_degenerada(w=%.4f,h=%.4f)" % (w, h)
    if w > 1.0 or h > 1.0:
        return "caixa_maior_que_a_imagem"

    return None


def validar_split(raiz: Path, split: str, nc: int, erros: list, vazios: list):
    pasta_img = raiz / split / "images"
    pasta_lbl = raiz / split / "labels"
    if not pasta_img.is_dir() or not pasta_lbl.is_dir():
        erros.append({"split": split, "arquivo": "-", "linha": "-",
                      "erro": "pasta_images_ou_labels_ausente"})
        return 0, 0

    imgs = {f.stem: f for f in pasta_img.iterdir() if f.suffix.lower() in EXTENSOES_IMG}
    lbls = {f.stem: f for f in pasta_lbl.glob("*.txt")}

    for stem in sorted(set(imgs) - set(lbls)):
        erros.append({"split": split, "arquivo": imgs[stem].name, "linha": "-",
                      "erro": "imagem_sem_rotulo"})
    for stem in sorted(set(lbls) - set(imgs)):
        erros.append({"split": split, "arquivo": lbls[stem].name, "linha": "-",
                      "erro": "rotulo_sem_imagem"})

    total_objetos = 0
    for stem, arq in sorted(lbls.items()):
        conteudo = arq.read_text(encoding="utf-8").strip()
        if not conteudo:
            vazios.append("%s/%s" % (split, arq.name))
            continue
        for i, linha in enumerate(conteudo.splitlines(), 1):
            if not linha.strip():
                continue
            motivo = validar_linha(linha, nc)
            if motivo:
                erros.append({"split": split, "arquivo": arq.name, "linha": str(i),
                              "erro": motivo})
            else:
                total_objetos += 1

    return len(imgs), total_objetos


def main():
    p = argparse.ArgumentParser(description="Validacao estrutural dos rotulos YOLO")
    p.add_argument("raiz", help="pasta do dataset (contem data.yaml)")
    p.add_argument("--saida", default="erros_estrutura.csv")
    args = p.parse_args()

    raiz = Path(args.raiz)
    data_yaml = raiz / "data.yaml"
    if not data_yaml.is_file():
        print("data.yaml nao encontrado em %s" % raiz)
        sys.exit(1)

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    nc = cfg["nc"]

    erros, vazios = [], []
    print("=" * 60)
    print("VALIDACAO ESTRUTURAL DOS ROTULOS")
    print("=" * 60)
    print("nc = %d  (%s)" % (nc, ", ".join(cfg["names"])))
    print()

    for split in SPLITS:
        n_imgs, n_obj = validar_split(raiz, split, nc, erros, vazios)
        print("%-6s %5d imagens, %5d objetos validos" % (split, n_imgs, n_obj))

    print()
    if vazios:
        print("Rotulos vazios (0 objetos, imagem so de fundo): %d" % len(vazios))
        for v in vazios[:10]:
            print("  %s" % v)
        if len(vazios) > 10:
            print("  ... e mais %d" % (len(vazios) - 10))
        print()

    if erros:
        with open(args.saida, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["split", "arquivo", "linha", "erro"])
            w.writeheader()
            w.writerows(erros)
        print("ERROS ESTRUTURAIS ENCONTRADOS: %d" % len(erros))
        for e in erros[:20]:
            print("  [%s] %s (linha %s): %s" % (e["split"], e["arquivo"], e["linha"], e["erro"]))
        if len(erros) > 20:
            print("  ... e mais %d, ver %s" % (len(erros) - 20, args.saida))
        print()
        print("RESULTADO: FALHA")
        sys.exit(1)
    else:
        print("RESULTADO: OK - nenhum erro estrutural encontrado")


if __name__ == "__main__":
    main()
