# Relatório de Qualidade e Distribuição das Classes

Dataset: `sows_farm_02.v2i.yolo26` — sistema anti-esmagamento de leitões em baia de maternidade
Sprint 3 — entregável "Relatório de qualidade e distribuição das classes"

## 1. Visão geral

| Split | Imagens |
|---|---|
| train | 833 |
| valid | 140 |
| test  | 107 |
| **Total** | **1080** |

Números após a correção de vazamento entre splits (ver seção 2). Dataset original do Roboflow tinha 1097 imagens.

## 2. Correção de vazamento entre splits

A checagem de duplicatas por hash perceptual (dHash) rodada sobre o dataset inteiro encontrou 46 duplicatas, das quais **17 cruzavam splits** — a mesma cena (frames da mesma câmera com 1–2 min de diferença) aparecia simultaneamente em train e valid/test, contaminando a avaliação.

Política aplicada: ao colidir train×valid/test, remove-se o lado train; ao colidir valid×test, remove-se o lado valid (test fica intocado, por ser o benchmark final). As 17 imagens + rótulos foram movidos para `sows_farm_02.v2i.yolo26/_quarentena_vazamento/` (não apagados). Detalhe por arquivo em `log_correcao_vazamento.csv`.

Efeito: train 848→833, valid 142→140, test 107→107.

## 3. Triagem de qualidade de imagem

Critérios: nitidez (variância do Laplaciano), brilho médio, contraste (desvio-padrão), resolução mínima (320px) e duplicata visual (dHash, tolerância 1 bit). Relatório completo por imagem em `relatorio_qualidade.csv`; resumo em `resumo_qualidade.txt`.

| Veredito | Imagens | % |
|---|---|---|
| Aproveitável | 1015 | 94,0% |
| Duvidosa | 7 | 0,6% |
| Descartar | 58 | 5,4% |

Motivos acumulados (uma imagem pode ter mais de um):

| Motivo | Imagens | % |
|---|---|---|
| Duplicata (dentro do mesmo split) | 31 | 2,9% |
| Borrada | 30 | 2,8% |
| Subexposta | 20 | 1,9% |
| Nitidez limítrofe | 7 | 0,6% |
| Baixo contraste | 1 | 0,1% |

**Pendência:** as 58 imagens "descartar" e 7 "duvidosa" ainda estão no dataset — só o vazamento entre splits foi corrigido até agora. Decisão de remover/mover essas 65 imagens fica para a geração do `dataset_v1.0/`.

## 4. Distribuição e balanceamento das classes

Contagem de instâncias (caixas), não de imagens — uma imagem pode ter várias. Detalhe em `distribuicao_classes.csv`.

| Classe | train | valid | test | total |
|---|---|---|---|---|
| piglet | 3528 | 641 | 472 | 4641 |
| sow-sleep-lactate | 460 | 93 | 60 | 613 |
| sow-sit | 131 | 15 | 13 | 159 |
| sow-sleep | 86 | 12 | 16 | 114 |
| sow-stand-feed | 79 | 9 | 8 | 96 |
| sow-stand | 77 | 11 | 10 | 98 |
| **Total** | **4361** | **781** | **579** | **5721** |

**Razão de desbalanceamento (classe mais comum / mais rara): 48,3x** (`piglet` vs `sow-stand`).

**Alertas** — classes abaixo de 20 instâncias em valid e/ou test (métrica por classe tende a ser instável/ruidosa nessas condições):
- `sow-sit`: valid 15, test 13
- `sow-sleep`: valid 12, test 16
- `sow-stand`: valid 11, test 10
- `sow-stand-feed`: valid 9, test 8

Ponto positivo: a proporção train/valid/test é consistente por classe (ex. `sow-stand`: 78,6%/11,2%/10,2%, próximo da proporção geral de imagens), ou seja, não há indício de que o split do Roboflow tenha sido enviesado — o desbalanceamento é do dataset como um todo, não um artefato da divisão.

## 5. Observações e recomendações

- O desbalanceamento (48x) é o maior risco de qualidade do dataset hoje — `piglet` domina; classes de comportamento da porca (`sow-stand`, `sow-stand-feed`, `sow-sleep`, `sow-sit`) têm poucos exemplos, especialmente em valid/test.
- Decisão já tomada para esta sprint: não fazer oversampling/augmentation direcionado agora — medir primeiro o efeito real do desbalanceamento no treinamento piloto, e só então decidir se é necessário.
- As 65 imagens "descartar"/"duvidosa" da triagem de qualidade (seção 3) ainda não foram removidas; isso deve ser resolvido antes de fechar o `dataset_v1.0/`.
- Nenhuma imagem corrompida foi encontrada (0 arquivos ilegíveis).

## Arquivos gerados

- `relatorio_qualidade.csv`, `resumo_qualidade.txt` — triagem de qualidade por imagem (atualizado pós-correção de vazamento)
- `distribuicao_classes.csv` — contagem de instâncias por classe/split
- `log_correcao_vazamento.csv` — imagens removidas do vazamento entre splits, com origem e destino
