# Experiments

- `zero-shot => without examples`
- `one-shot => with one example`

### e1. zero-shot | subgraph triple list to dialogue generation.
-  PROMPT-1: "Generate a list of n questions based on a subgraph from a knowledge graph, represented as a list of triplets. Each question should relate to a shared entity (e) within the subgraph and should fall into one of the following categories: list, count, boolean, wh (open-ended), or date-related questions. Each question should be answerable solely from the information in the provided subgraph without explicitly mentioning it. The questions can be equivalent to one or two triplets from the subgraph."
- no example

### e2. one-shot | subgraph triple list to dialogue generation.
- PROMPT-1 (same as e1) + with example
- example is of format
  - input: [(bob,studentof,concordia),(),...]
  - output [q1,q2,q3]

### e3. zero-shot | [seed + schema info] to dialogue generation.
- PROMPT-1: "Generate a list of n questions based on a subgraph **`schema`**  from a knowledge graph ~~represented as a list of triplets~~. Each question shoul`d relate to a shared entity (e) within the subgraph and should fall into one of the following categories: list, count, boolean, wh (open-ended), or date-related questions. Each question should be answerable solely from the information in the provided subgraph without explicitly mentioning it. The questions can be equivalent to one or two triplets from the subgraph."
- no example

### e4. one-shot | [seed + schema info] to dialogue generation.




---

### analysis

#### llm output parse eror
|experiment|parse_error|success|total|
|-|-|-|-|
|e1|25|25|50|
|e2|22|28|50|
|e3|30|20|50|
|e4|||50|


#### question type count
|experiment|howmany|what|is|when|who|other|total|
|-|-|-|-|-|-|-|-|
|e1|25|45|15|17|7|16||
|e2|27|31|11|5|2|69||
|e3|26|41|17|9|25|12||
|e4||||||||


#### llm tokens = cost analysis

| exp1    |   total_tokens |   prompt_tokens |   completion_tokens |
|:--------|---------------:|----------------:|--------------------:|
| Average |        1293.76 |          1045.4 |              248.36 |
| Minimum |        1038    |           852   |              181    |
| Maximum |        1570    |          1316   |              430    |
| Total   |       32344    |         26135   |             6209    |

| exp2    |   total_tokens |   prompt_tokens |   completion_tokens |
|:--------|---------------:|----------------:|--------------------:|
| Average |        1829.83 |         1562.59 |             267.241 |
| Minimum |        1599    |         1390    |             199     |
| Maximum |        2111    |         1844    |             478     |
| Total   |       53065    |        45315    |            7750     |

| exp3    |   total_tokens |   prompt_tokens |   completion_tokens |
|:--------|---------------:|----------------:|--------------------:|
| Average |        1231.58 |         937.538 |             294.038 |
| Minimum |        1006    |         838     |             167     |
| Maximum |        1426    |        1047     |             418     |
| Total   |       32021    |       24376     |            7645     |
---

### examples
```
"What year was Yosra Zguira's publication 'Study and development of wireless sensor network architecture tolerant to delays. (Etude et d\u00e9veloppement d'une architecture de r\u00e9seaux de capteurs tol\u00e9rante aux d\u00e9lais). (2018)' published?"
```
- the question does have answer in it self, its not using predicate to get the question, its using the partial description of subject.


```
"Is Andreas Geiger 0002 the only author of the paper 'Bedienung digitaler Menschmodelle zur Absicherung manueller Montaget\u00e4tigkeiten durch Virtual-Reality-Interaktion.' in 2021?"
```

- can kgqa system answer such questions?

