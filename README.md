# livetx — transcrição ao vivo de streams

Captura o áudio de uma live (YouTube, Instagram, G1/Globoplay, Twitch, HLS, RTMP, UDP…), transcreve
em português com [faster-whisper](https://github.com/SYSTRAN/faster-whisper) na GPU e grava em log.

```
URL ──yt-dlp/streamlink──▶ URL de mídia ──ffmpeg──▶ PCM 16 kHz ──VAD (Silero)──▶ segmentos ──Whisper──▶ logs/<stream>/<data>.log
```

## Instalação
```
pixi install
pixi run check-cuda      # deve imprimir 1 (GPU visível)
```

## Uso
```
pixi run tx "https://www.youtube.com/@CNNbrasil/live"
pixi run tx "https://www.instagram.com/<usuario>/live/" --cookies-from-browser firefox
pixi run tx "https://exemplo.com/stream/index.m3u8" --model medium
tail -f logs/<stream>/$(date +%F).log
```
Formato do log: `[2026-10-22 21:50:03] P1: texto transcrito…` (horário local do início da fala; um arquivo por dia).

No terminal, cada falante aparece numa cor (a cor segue o rótulo P1, P2…, e continua a mesma quando ele ganha nome
no `people.txt`). As mensagens do sistema saem em itálico, em stderr: `ⓘ` informação, `⚠` aviso, `✖` erro. Com
pipe/redirecionamento, a saída vira texto puro, uma linha por fala, igual ao arquivo de log. `--color never` (ou
`NO_COLOR=1`) desliga as cores; `--quiet` esconde a transcrição no terminal; `-v` mostra também as mensagens de
depuração e as das bibliotecas.

Opções principais (`pixi run tx --help`):

| Opção | Padrão | |
|---|---|---|
| `--model` | `small` | `tiny`/`base` (mais rápido), `medium`/`turbo` (melhor; cabem nos 6 GB) |
| `--device` | `auto` | `cuda` ou `cpu` |
| `--resolver` | `yt-dlp` | `streamlink` (bom p/ Twitch), `direct` (URL já é a mídia) |
| `--cookies`, `--cookies-from-browser` | — | lives que exigem login (Instagram) |
| `--max-segment` | `15` | duração máxima de um segmento, em segundos |
| `--max-backlog` | `4` | segmentos em espera antes de descartar áudio, para não acumular atraso |
| `--name`, `--out-dir` | derivado / `logs` | onde gravar |

A stream é re-resolvida a cada queda (as URLs do YouTube expiram), com backoff de até 60 s. Ctrl+C encerra
e transcreve o que restou no buffer.

Latência: um segmento fecha após ~0,5 s de pausa (ou após `--max-segment`); some ~0,5–1,5 s de inferência.
Conteúdo com DRM (parte do Globoplay) não pode ser capturado.

## Falantes (P1, P2, … PN)
Cada linha leva o falante: `[2026-10-22 22:06:14] P1: Fábio, o que está acontecendo com a meta?`.
Se a voz muda no meio de um trecho, a linha é dividida na palavra da troca.

Funciona por embeddings de voz (3D-Speaker CAM++ via sherpa-onnx, CPU, ~20 ms por janela, baixado no 1º uso
para `~/.cache/livetx/models/`). As janelas de 1,5 s são comparadas por cosseno com as vozes conhecidas. Uma voz
nova só é criada após ~3 s de fala seguida sem correspondência; trechos mais curtos (vinheta, música, um "ok")
ficam com a voz conhecida mais parecida ou saem como `?`. A troca de falante exige vantagem de 0,1 sobre o
atual. As vozes ficam em `logs/<stream>/speakers.npz` e mantêm o número entre execuções
(`--reset-speakers` para esquecer). Se o modelo de voz mudar, o arquivo antigo é renomeado
(`speakers.<modelo>.npz`) e as vozes começam do zero, porque vetores de modelos diferentes não se comparam.

No confronto direto do debate da Band (13–25 min de https://www.youtube.com/watch?v=RugVB17uRVM), 95% das
palavras ficaram com o falante certo (o modelo anterior, WeSpeaker ResNet34, acertava 81%). Os erros
restantes estão nas fronteiras de turno (~1 s) e em trocas muito rápidas.

**Nomes:** o operador escreve num `people.txt` (relido a cada 10 s; outro caminho com `--people`):
```
P1:William Waack
P2:Claudio
```
A partir daí, as linhas novas saem com o nome (`[…] William Waack: …`); as já gravadas não mudam. Linhas vazias e
`# comentários` são ignoradas. Para várias lives em paralelo, use um `--people` por processo, porque os números são
por stream.

Ajuste fino: `--speaker-threshold` (padrão 0,45). Mais alto separa mais e cria falantes espúrios; mais baixo funde
vozes. `-v` mostra a similaridade de cada janela. `--no-diarize` desliga a identificação de falantes.

Limitações (observadas):
- Vozes muito parecidas no mesmo estúdio/microfone ainda podem ser fundidas num só Pn.
- Fala sobreposta fica com o falante dominante; réplicas muito curtas (< ~1,5 s) ficam com o vizinho.
- A mesma pessoa em outro microfone pode virar um Pn novo. Nesse caso, dê o mesmo nome aos dois no `people.txt`.

## Testes
```
pixi run test
# simulação de live local:
pixi run ffmpeg -re -i amostra.mp3 -f mpegts udp://127.0.0.1:1234 &
pixi run tx udp://127.0.0.1:1234
```
