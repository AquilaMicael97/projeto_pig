import argparse
import csv
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Metricas por imagem

def nitidez(cinza: np.ndarray) -> float:
    """Variancia do Laplaciano. Quanto menor, mais borrada a imagem."""
    return float(cv2.Laplacian(cinza, cv2.CV_64F).var())


def hash_perceptual(cinza: np.ndarray) -> int:
    """
    dHash de 64 bits. Redimensiona para 9x8 e compara cada pixel com o
    vizinho da direita. Robusto a mudancas de escala e a compressao JPEG.
    """
    pequena = cv2.resize(cinza, (9, 8), interpolation=cv2.INTER_AREA)
    diff = pequena[:, 1:] > pequena[:, :-1]
    valor = 0
    for bit in diff.flatten():
        valor = (valor << 1) | int(bit)
    return valor


def analisar(caminho: Path):
    """Retorna o dicionario de metricas de uma imagem, ou None se ilegivel."""
    img = cv2.imread(str(caminho))
    if img is None:
        return None

    altura, largura = img.shape[:2]
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return {
        "arquivo": caminho.name,
        "caminho": str(caminho),
        "largura": largura,
        "altura": altura,
        "nitidez": round(nitidez(cinza), 2),
        "brilho": round(float(cinza.mean()), 2),
        "contraste": round(float(cinza.std()), 2),
        "_hash": hash_perceptual(cinza),
    }

# Deteccao de duplicatas  (indexacao multi-segmento)

def distancia_hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _segmentos(h: int, nseg: int, bits: int = 64):
    """Divide o hash em nseg pedacos de tamanho aproximadamente igual."""
    tamanhos = [bits // nseg + (1 if i < bits % nseg else 0) for i in range(nseg)]
    saida, desloc = [], 0
    for t in tamanhos:
        saida.append((h >> desloc) & ((1 << t) - 1))
        desloc += t
    return saida


def marcar_duplicatas(registros: list, tolerancia: int) -> int:
    """
    Marca duplicatas mantendo a primeira ocorrencia de cada grupo.

    Usa o principio da casa dos pombos: se dois hashes de 64 bits diferem em
    no maximo D bits, entao ao dividi-los em D+1 segmentos, pelo menos um
    segmento tem de ser identico. Assim basta comparar candidatos que
    coincidem em algum segmento, em vez de comparar todos contra todos.
    O resultado e exato, nao aproximado, e torna viavel rodar em dezenas
    de milhares de imagens.
    """
    nseg = tolerancia + 1
    indices = [defaultdict(list) for _ in range(nseg)]
    total = 0

    for pos, r in enumerate(registros):
        segs = _segmentos(r["_hash"], nseg)

        candidatos = set()
        for i, s in enumerate(segs):
            candidatos.update(indices[i].get(s, ()))

        achou = None
        for c in candidatos:
            if distancia_hamming(r["_hash"], registros[c]["_hash"]) <= tolerancia:
                achou = registros[c]["arquivo"]
                break

        if achou:
            r["duplicata_de"] = achou
            total += 1
        else:
            r["duplicata_de"] = ""
            for i, s in enumerate(segs):
                indices[i][s].append(pos)

    return total

# Classificacao

def classificar(m: dict, lim: dict):
    """Aplica os limiares e devolve (veredito, motivos)."""
    motivos = []

    if m["largura"] < lim["resolucao"] or m["altura"] < lim["resolucao"]:
        motivos.append("resolucao_baixa")

    if m["nitidez"] < lim["blur"]:
        motivos.append("borrada")
    elif m["nitidez"] < lim["blur"] * 2:
        motivos.append("nitidez_limitrofe")

    if m["brilho"] < lim["escuro"]:
        motivos.append("subexposta")
    elif m["brilho"] > lim["claro"]:
        motivos.append("superexposta")

    if m["contraste"] < lim["contraste"]:
        motivos.append("baixo_contraste")

    if m.get("duplicata_de"):
        motivos.append("duplicata")

    graves = {"borrada", "subexposta", "superexposta", "duplicata", "resolucao_baixa"}
    if any(x in graves for x in motivos):
        veredito = "descartar"
    elif motivos:
        veredito = "duvidosa"
    else:
        veredito = "aproveitavel"

    return veredito, ";".join(motivos) if motivos else "-"


# Relatorios

def histograma(valores: list, rotulo: str, faixas: int = 12) -> str:
    if not valores:
        return ""
    vmin, vmax = min(valores), max(valores)
    if vmax == vmin:
        return "%s: todos os valores iguais a %.1f\n" % (rotulo, vmin)

    largura = (vmax - vmin) / faixas
    contagem = [0] * faixas
    for v in valores:
        idx = min(int((v - vmin) / largura), faixas - 1)
        contagem[idx] += 1

    pico = max(contagem) or 1
    linhas = ["%s  (min %.1f / max %.1f)" % (rotulo, vmin, vmax)]
    for i, c in enumerate(contagem):
        ini = vmin + i * largura
        barra = "#" * int(46 * c / pico)
        linhas.append("  %10.1f | %-46s %d" % (ini, barra, c))
    return "\n".join(linhas) + "\n"


def gerar_resumo(registros: list, lim: dict, dup: int, pasta: str) -> str:
    total = len(registros)
    contagem = Counter(r["veredito"] for r in registros)
    motivos = Counter()
    for r in registros:
        if r["motivos"] != "-":
            for mo in r["motivos"].split(";"):
                motivos[mo] += 1

    linhas = [
        "=" * 66,
        "TRIAGEM DE QUALIDADE - RESULTADO",
        "=" * 66,
        "Pasta analisada: %s" % pasta,
        "Imagens analisadas: %d" % total,
        "",
        "VEREDITO",
    ]
    for nome in ("aproveitavel", "duvidosa", "descartar"):
        n = contagem.get(nome, 0)
        pct = (100.0 * n / total) if total else 0.0
        linhas.append("  %-14s %7d   (%5.1f%%)" % (nome, n, pct))

    linhas += ["", "MOTIVOS (uma imagem pode acumular mais de um)"]
    if motivos:
        for mo, n in motivos.most_common():
            pct = 100.0 * n / total if total else 0.0
            linhas.append("  %-22s %7d   (%5.1f%%)" % (mo, n, pct))
    else:
        linhas.append("  nenhum")

    unicas = total - dup
    linhas += [
        "",
        "REDUNDANCIA",
        "  duplicatas encontradas   %7d" % dup,
        "  imagens visualmente unicas %5d   (%5.1f%%)" % (
            unicas, (100.0 * unicas / total) if total else 0.0),
        "",
        "LIMIARES UTILIZADOS",
        "  nitidez minima (Laplaciano)  %s" % lim["blur"],
        "  brilho aceitavel             %s a %s" % (lim["escuro"], lim["claro"]),
        "  contraste minimo             %s" % lim["contraste"],
        "  resolucao minima             %s px" % lim["resolucao"],
        "  tolerancia de duplicata      %s bits" % lim["dup"],
        "",
        "=" * 66,
        "DISTRIBUICOES  (use para calibrar os limiares acima)",
        "=" * 66,
        "",
        histograma([r["nitidez"] for r in registros], "Nitidez"),
        histograma([r["brilho"] for r in registros], "Brilho medio"),
        histograma([r["contraste"] for r in registros], "Contraste"),
    ]
    return "\n".join(linhas)


# Programa principal

def main():
    p = argparse.ArgumentParser(
        description="Triagem de qualidade de dataset de imagens")
    p.add_argument("pasta", help="pasta com as imagens (percorre subpastas)")
    p.add_argument("--saida", default=".", help="pasta para salvar os relatorios")
    p.add_argument("--copiar", action="store_true",
                   help="copia as imagens aprovadas para <saida>/aprovadas/")
    p.add_argument("--blur", type=float, default=100.0,
                   help="variancia minima do Laplaciano (padrao 100)")
    p.add_argument("--escuro", type=float, default=35.0,
                   help="brilho medio minimo (padrao 35)")
    p.add_argument("--claro", type=float, default=225.0,
                   help="brilho medio maximo (padrao 225)")
    p.add_argument("--contraste", type=float, default=18.0,
                   help="desvio-padrao minimo do brilho (padrao 18)")
    p.add_argument("--resolucao", type=int, default=320,
                   help="menor lado aceitavel em px (padrao 320)")
    p.add_argument("--dup", type=int, default=1,
                   help="distancia de Hamming para considerar duplicata (padrao 5)")
    args = p.parse_args()

    base = Path(args.pasta)
    if not base.is_dir():
        print("Pasta nao encontrada: %s" % base)
        sys.exit(1)

    arquivos = sorted(f for f in base.rglob("*")
                      if f.suffix.lower() in EXTENSOES and f.is_file())
    if not arquivos:
        print("Nenhuma imagem encontrada em %s" % base)
        sys.exit(1)

    print("Analisando %d imagens..." % len(arquivos))

    registros, ilegiveis = [], []
    for i, caminho in enumerate(arquivos, 1):
        if i % 500 == 0:
            print("  %d/%d" % (i, len(arquivos)))
        m = analisar(caminho)
        if m is None:
            ilegiveis.append(str(caminho))
        else:
            registros.append(m)

    lim = {"blur": args.blur, "escuro": args.escuro, "claro": args.claro,
           "contraste": args.contraste, "resolucao": args.resolucao,
           "dup": args.dup}

    print("Procurando duplicatas...")
    dup = marcar_duplicatas(registros, args.dup)

    for r in registros:
        r["veredito"], r["motivos"] = classificar(r, lim)

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    csv_path = saida / "relatorio_qualidade.csv"
    campos = ["arquivo", "veredito", "motivos", "nitidez", "brilho",
              "contraste", "largura", "altura", "duplicata_de", "caminho"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for r in registros:
            w.writerow(r)

    resumo = gerar_resumo(registros, lim, dup, str(base))
    if ilegiveis:
        resumo += "\nARQUIVOS ILEGIVEIS: %d\n" % len(ilegiveis)
        for x in ilegiveis[:20]:
            resumo += "  %s\n" % x

    txt_path = saida / "resumo_qualidade.txt"
    txt_path.write_text(resumo, encoding="utf-8")

    copiadas = 0
    if args.copiar:
        destino = saida / "aprovadas"
        destino.mkdir(parents=True, exist_ok=True)
        for r in registros:
            if r["veredito"] == "aproveitavel":
                shutil.copy2(r["caminho"], destino / r["arquivo"])
                copiadas += 1

    print()
    print(resumo)
    print("CSV detalhado: %s" % csv_path)
    print("Resumo:        %s" % txt_path)
    if args.copiar:
        print("Aprovadas:     %s  (%d arquivos)" % (saida / "aprovadas", copiadas))


if __name__ == "__main__":
    main()