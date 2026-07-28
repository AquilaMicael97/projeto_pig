#!/usr/bin/env python3
"""
Correcao de vazamento de dados entre splits (train/valid/test)
Projeto: sistema anti-esmagamento de leitoes em baia de maternidade

CONTEXTO
    triagem_qualidade_dataset.py ja identifica duplicatas (dHash) rodando
    sobre o dataset inteiro de uma vez, o que por acaso ja compara imagens
    de splits diferentes entre si. Este script le o relatorio_qualidade.csv
    ja gerado e isola so os pares de duplicata que CRUZAM splits — ou seja,
    onde a mesma cena aparece em mais de um split (train/valid/test).

POLITICA DE REMOCAO
    - Se o par envolve "train" e ("valid" ou "test"): remove o lado train.
      Train e o split grande, perder algumas dezenas de imagens nao dói.
    - Se o par e "valid" vs "test": remove o lado valid.
      Test e o benchmark final, deve ficar intocado.
    - Nada e apagado: as imagens e rotulos vao para uma pasta de
      quarentena, preservando de qual split vieram.

USO
    python corrigir_vazamento.py
    python corrigir_vazamento.py --csv relatorio_qualidade.csv --raiz sows_farm_02.v2i.yolo26
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

SPLITS = ("train", "valid", "test")


def split_de(caminho: str) -> str:
    partes = Path(caminho).parts
    for s in SPLITS:
        if s in partes:
            return s
    return "?"


def rotulo_de(caminho_imagem: Path) -> Path:
    """Mesmo nome, pasta labels irma de images."""
    return caminho_imagem.parent.parent / "labels" / (caminho_imagem.stem + ".txt")


def decidir_remocao(linha: dict, por_arquivo: dict):
    """
    Devolve o registro (linha do CSV) a remover para este par de duplicata,
    ou None se o par nao cruza splits ou nao se aplica a politica.
    """
    dup_nome = linha["duplicata_de"]
    if not dup_nome or dup_nome not in por_arquivo:
        return None

    original = por_arquivo[dup_nome]
    s1 = split_de(linha["caminho"])
    s2 = split_de(original["caminho"])
    if s1 == s2:
        return None  # duplicata dentro do mesmo split, fora do escopo aqui

    if "train" in (s1, s2):
        return linha if s1 == "train" else original
    if {s1, s2} == {"valid", "test"}:
        return linha if s1 == "valid" else original
    return None


def main():
    p = argparse.ArgumentParser(description="Corrige vazamento de dados entre splits")
    p.add_argument("--csv", default="relatorio_qualidade.csv")
    p.add_argument("--raiz", default="sows_farm_02.v2i.yolo26")
    p.add_argument("--quarentena", default=None,
                   help="pasta de quarentena (padrao: <raiz>/_quarentena_vazamento)")
    args = p.parse_args()

    csv_path = Path(args.csv)
    raiz = Path(args.raiz)
    if not csv_path.is_file():
        print("CSV nao encontrado: %s" % csv_path)
        sys.exit(1)

    quarentena = Path(args.quarentena) if args.quarentena else raiz / "_quarentena_vazamento"

    linhas = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    por_arquivo = {r["arquivo"]: r for r in linhas}

    a_remover = {}  # arquivo -> (linha, split_origem, par_com, split_par)
    for linha in linhas:
        alvo = decidir_remocao(linha, por_arquivo)
        if alvo is None:
            continue
        outro = por_arquivo[linha["duplicata_de"]]
        par = outro if alvo is linha else linha
        a_remover[alvo["arquivo"]] = (
            alvo, split_de(alvo["caminho"]), par["arquivo"], split_de(par["caminho"]))

    if not a_remover:
        print("Nenhum vazamento entre splits encontrado. Nada a fazer.")
        return

    log = []
    movidas, faltando = 0, []
    for arquivo, (linha, split_origem, par_arquivo, split_par) in a_remover.items():
        img_path = Path(linha["caminho"])
        rot_path = rotulo_de(img_path)

        destino_img = quarentena / split_origem / "images" / img_path.name
        destino_rot = quarentena / split_origem / "labels" / rot_path.name
        destino_img.parent.mkdir(parents=True, exist_ok=True)
        destino_rot.parent.mkdir(parents=True, exist_ok=True)

        if not img_path.is_file():
            faltando.append(str(img_path))
            continue

        shutil.move(str(img_path), str(destino_img))
        if rot_path.is_file():
            shutil.move(str(rot_path), str(destino_rot))
        movidas += 1
        log.append({
            "arquivo_removido": arquivo,
            "split_origem": split_origem,
            "duplicata_de": par_arquivo,
            "split_duplicata": split_par,
        })

    log_path = raiz.parent / "log_correcao_vazamento.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arquivo_removido", "split_origem",
                                          "duplicata_de", "split_duplicata"])
        w.writeheader()
        w.writerows(log)

    print("=" * 60)
    print("CORRECAO DE VAZAMENTO ENTRE SPLITS")
    print("=" * 60)
    print("Pares cruzando splits detectados: %d" % len(a_remover))
    print("Imagens movidas para quarentena:  %d" % movidas)
    if faltando:
        print("Arquivos que ja nao existiam:      %d" % len(faltando))
        for f in faltando:
            print("  %s" % f)
    print("Quarentena: %s" % quarentena.resolve())
    print("Log:        %s" % log_path.resolve())

    print()
    print("Contagem de imagens por split apos a correcao:")
    for s in SPLITS:
        pasta = raiz / s / "images"
        n = len(list(pasta.glob("*"))) if pasta.is_dir() else 0
        print("  %-6s %d" % (s, n))


if __name__ == "__main__":
    main()
