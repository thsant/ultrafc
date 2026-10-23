<img src="docs/mascot-512px.png" alt="Mascote do ultrafc: um carcará detetive, com chapéu de Sherlock Holmes e lupa" width="160" align="right">

# ultrafc — transcrição, falantes, checagem de fatos e publicação de lives

O **ultrafc** acompanha uma transmissão em vídeo (YouTube, Instagram, G1/Globoplay, Twitch, HLS, RTMP, UDP…)
e cobre quatro etapas:

1. **Transcrição** ao vivo em português, com [faster-whisper](https://github.com/SYSTRAN/faster-whisper) na GPU;
2. **Identificação dos falantes** por embeddings de voz (P1, P2… ou nomes informados pelo operador);
3. **Checagem de fatos** com LLMs de raciocínio (Claude): um modelo encontra as afirmações verificáveis e
   agentes com busca na web as checam em paralelo;
4. **Publicação** das checagens e de resumos num canal do **Telegram**, sempre com aprovação de um operador.

![Fluxo do ultrafc: pixi run tx transcreve e identifica falantes; pixi run check detecta e checa afirmações; pixi run publish envia checagens e resumos para revisão e publica no Telegram](docs/pipeline.svg)

Cada etapa é um processo separado que lê a saída da anterior; dá para usar só a transcrição.

![Dois terminais durante o debate da Band: à esquerda, a transcrição com o horário e o nome de cada falante; à direita, a checagem de fatos com as afirmações detectadas, as checagens em andamento e os resultados coloridos (verdadeiro, omissão, exagerado, inconclusivo)](docs/screenshot.png)

*Debate da Band ([vídeo](https://www.youtube.com/watch?v=RugVB17uRVM)): à esquerda, `pixi run tx` com os falantes já
nomeados pelo `people.txt`; à direita, `pixi run check` detectando afirmações e checando até 3 em paralelo.*

## Instalação
```
pixi install
pixi run check-cuda      # deve imprimir 1 (GPU visível)
```

## Uso
```
pixi run tx "https://www.youtube.com/watch?v=RugVB17uRVM"     # debate presidencial da Band, 2º turno
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
Cada linha leva o falante: `[2026-10-22 22:12:41] P4: Você não respondeu à pergunta.`.
Se a voz muda no meio de um trecho, a linha é dividida na palavra da troca.

Funciona por embeddings de voz (3D-Speaker CAM++ via sherpa-onnx, CPU, ~20 ms por janela, baixado no 1º uso
para `~/.cache/ultrafc/models/`). As janelas de 1,5 s são comparadas por cosseno com as vozes conhecidas. Uma voz
nova só é criada após ~3 s de fala seguida sem correspondência; trechos mais curtos (vinheta, música, um "ok")
ficam com a voz conhecida mais parecida ou saem como `?`. A troca de falante exige vantagem de 0,1 sobre o
atual. As vozes ficam em `logs/<stream>/speakers.npz` e mantêm o número entre execuções
(`--reset-speakers` para esquecer). Se o modelo de voz mudar, o arquivo antigo é renomeado
(`speakers.<modelo>.npz`) e as vozes começam do zero, porque vetores de modelos diferentes não se comparam.

No confronto direto do debate da Band (13–25 min do vídeo do exemplo), 95% das
palavras ficaram com o falante certo (o modelo anterior, WeSpeaker ResNet34, acertava 81%). Os erros
restantes estão nas fronteiras de turno (~1 s) e em trocas muito rápidas.

**Nomes:** o operador escreve num `people.txt` (relido a cada 10 s; outro caminho com `--people`):
```
P1:Eduardo Oinegue
P2:Adriana Araujo
```
A partir daí, as linhas novas saem com o nome (`[…] Eduardo Oinegue: …`); as já gravadas não mudam. Linhas vazias e
`# comentários` são ignoradas. Para várias lives em paralelo, use um `--people` por processo, porque os números são
por stream.

Ajuste fino: `--speaker-threshold` (padrão 0,45). Mais alto separa mais e cria falantes espúrios; mais baixo funde
vozes. `-v` mostra a similaridade de cada janela. `--no-diarize` desliga a identificação de falantes.

Limitações (observadas):
- Vozes muito parecidas no mesmo estúdio/microfone ainda podem ser fundidas num só Pn.
- Fala sobreposta fica com o falante dominante; réplicas muito curtas (< ~1,5 s) ficam com o vizinho.
- A mesma pessoa em outro microfone pode virar um Pn novo. Nesse caso, dê o mesmo nome aos dois no `people.txt`.

## Checagem de fatos
Um processo separado acompanha o log e usa o Claude para encontrar e checar afirmações:

```
pixi run check --log-dir logs/<stream>        # em outro terminal, junto com o pixi run tx
```
1. **Detecção** (`claude-sonnet-5-5`, com raciocínio): a cada 6 linhas novas (ou 45 s), analisa o trecho com
   10 linhas de contexto e grava cada afirmação verificável em `checking/to_check/<id>.md` (afirmação, quem
   disse, horário, fala literal e por que checar).
2. **Checagem** (`claude-haiku-5-5` com busca e leitura de páginas na web): até 3 agentes em paralelo
   (`--max-parallel`), um por afirmação; os demais esperam na fila. Uma fonte citada pelo agente que não
   apareceu na pesquisa dele é marcada *(não consultada pelo agente)*. Uma falha gera `resultado: erro`
   com o motivo (vai para `checked/`, mas não é publicada).
3. **Revisão** (`claude-opus-5-5`, `--review-model`): confere o veredito relendo só as fontes que o
   checador citou, sem pesquisar outras (até 5 mil tokens por página). Aprovado, vai para
   `checking/checked/<id>.md` (resultado, justificativa, fontes e a nota da revisão); reprovado, ou se a
   revisão falhar, vai para `checking/rejected/<id>.md` e nunca é publicado.

   | Resultado | Quando |
   |---|---|
   | **verdadeiro** | as fontes confirmam o essencial (detalhes secundários vão na justificativa) |
   | **falso** | as fontes contradizem o essencial |
   | **exagerado** | o fato existe, mas o número, a intensidade ou o alcance foi inflado |
   | **omissão** | o que foi dito está correto, mas falta um contexto que muda a conclusão |
   | **inconclusivo** | depois da pesquisa, as fontes não bastam para confirmar nem refutar |

Precisa de `ANTHROPIC_API_KEY` no ambiente (exportada no topo do `~/.bashrc`, antes do bloco que
interrompe shells não interativos). Por padrão só analisa o que for escrito depois de iniciar; `--from-start`
analisa o log todo, `--once` sai ao terminar, e `--stop-after N` limita o número de afirmações (útil para
testar com custo controlado). Ao reiniciar, continua de onde parou e retoma as afirmações sem resultado.
O uso da API (tokens, buscas e custo estimado por modelo) é gravado a cada checagem em
`checking/usage/<stream>-<início>.json`, e o do publish em `checking/usage/publish-<início>.json`.

É uma triagem automática: os vereditos devem ser revisados por uma pessoa antes de qualquer publicação.

## Publicação no Telegram
Um terceiro processo leva as checagens e um resumo periódico a um canal do Telegram, **sempre com aprovação
de um operador**: cada item chega antes a um chat de revisão com os botões ✅ Publicar / ❌ Descartar, e só
vai ao canal depois do clique.

Configuração (uma vez):
1. No Telegram, fale com o **@BotFather** → `/newbot` e guarde o token. Ponha no **topo** do `~/.bashrc`:
   `export TELEGRAM_BOT_TOKEN="..."`.
2. Crie o canal e adicione o bot como **administrador** com permissão de publicar mensagens.
3. Abra uma conversa com o bot (ou crie um grupo da equipe e adicione o bot): esse é o chat de revisão.
4. `pixi run publish --whoami` e mande uma mensagem ao bot nesse chat: o terminal mostra o id do chat
   (`--review-chat`) e o seu id de usuário (`--operators`).

Uso, ao lado do `tx` e do `check`:
```
pixi run publish --channel @meu_canal --review-chat <id> --operators <id1,id2> --log-dir logs/<stream> \
    --footer "Checagem assistida por IA, revisada pela equipe"
```
- **Checagens:** cada arquivo novo em `checking/checked/` vira um rascunho (resultado com ícone, quem,
  afirmação, justificativa resumida e até 5 fontes consultadas). Resultados `erro` não vão para revisão.
- **Cards:** antes do texto vai um card quadrado (1080×1080 PNG), pronto para Instagram e WhatsApp: faixa na
  cor do resultado (verde verdadeiro, vermelho falso, laranja exagerado, amarelo omissão, cinza
  inconclusivo), quem disse, a afirmação, o resumo de 1–2 frases escrito pelo checador e revisado, os
  domínios das fontes e, no rodapé, o @nome do canal (`--card-brand` troca) e o `--footer`. Os PNGs ficam
  em `checking/cards/`. `--no-cards` publica só o texto.
- **Resumos:** a cada `--summary-minutes` (padrão 10), com pelo menos 8 falas novas, o `claude-sonnet-5-5`
  escreve um resumo neutro em tópicos, atribuído aos falantes; fica em `checking/summaries/` e também passa
  pela revisão. `--no-summary` desliga.
- Só os ids em `--operators` podem aprovar. O que foi enviado, publicado ou descartado fica em
  `checking/.publish-state.json`: um reinício não reenvia nem publica duas vezes.

## Áudio do computador
Se a stream não abre (yt-dlp/streamlink falham), abra a transmissão no navegador e transcreva o som que sai
no computador:
```
pixi run tx pulse: --name <stream>
```
`pulse:` usa o monitor da saída de áudio padrão (PulseAudio/PipeWire), não o microfone. Para outra fonte,
`pixi run tx pulse:<fonte>` com um nome de `pactl list short sources` terminado em `.monitor`. Repetir o
`--name` da stream original mantém a pasta de log, os falantes salvos e a checagem em andamento. Deixe o
navegador sem mudo e evite outros sons no computador, que também seriam transcritos.

## Testes
```
pixi run test
# simulação de live local:
pixi run ffmpeg -re -i amostra.mp3 -f mpegts udp://127.0.0.1:1234 &
pixi run tx udp://127.0.0.1:1234
```

## Licença
[MIT](LICENSE) — uso, cópia, modificação e redistribuição livres, inclusive comercial, mantendo o aviso de copyright.
