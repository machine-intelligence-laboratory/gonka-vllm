# TOPLOC outlier analysis — A100↔H100 FP8 false positives

FPR budget: **0.50%**   |   threshold: learned linear `t(L) = a·log₁₀(seq_len) + b + lift`   |   configs analyzed: **49** (7×7 grid of k, bs)

Metric: per-sample mean `mant_err_mean`. A sample is an FP if its mean exceeds the learned threshold at its own seq_len.

## FP persistence across the 49 configs

| direction | #unique FP samples | max configs flagging one sample |
|---|---|---|
| A100→H100 | 33 | 48 |
| H100→A100 | 25 | 49 |

## Most persistent false positives

| sample idx | A100→H100 flags | H100→A100 flags | A100 seq_len | H100 seq_len | language | prompt (truncated) |
|---:|---:|---:|---:|---:|:---:|---|
| 605 | 48/49 | 49/49 | 3152 | 3041 | hi | जेवीएम का "कचरा संग्रह" लॉग निम्नानुसार स्वरूपित है: \n 1. प्रत्येक पंक्ति टाइमस्टैम्प से शुरू होने वाले एकल कचरा संग्रह संचालन का प्रतिनिधित्व करती है। \n 2. जेवीएम में विभिन्न मे… |
| 692 | 45/49 | 49/49 | 227 | 186 | hi | मुझे नीचे दी गई कहानी का एक पंक्ति का सारांश दें: \n  \n बहुत समय पहले उड़ीसा की रानी ने सुना था कि एक प्रस्तर मूर्तिकार था, जिसने देवताओं की ऐसी अद्भुत मूर्तियाँ बनाई थीं कि लोग म… |
| 23 | 29/49 | 17/49 | 33 | 33 | en | Transcribe the recording into text. |
| 1009 | 6/49 | 29/49 | 83 | 83 | en | Compare the two given products and provide the advantages and disadvantages. |
| 534 | 11/49 | 11/49 | 7 | 7 | ch | 写一个五个词的句子来描述你的一天。 |
| 823 | 11/49 | 11/49 | 6 | 6 | ar | استخدم كلمة واحدة لوصف الطقس اليوم. |
| 97 | 20/49 | 1/49 | 128 | 116 | en | Compute the area of a rectangle with length 10cm and width 5cm. |
| 149 | 0/49 | 17/49 | 54 | 54 | en | Calculate the square root of a given number. |
| 144 | 0/49 | 14/49 | 489 | 320 | en | Find the cutoff score for successful completion of the exam |
| 55 | 0/49 | 14/49 | 179 | 176 | en | What is the force on a 1 kg mass due to the gravitational force? |

## Deep dive: the two dominant outliers

### Sample idx = 605
- **FP persistence**: A100→H100 48/49, H100→A100 49/49
- **Language**: hi
- **Prompt** (first 500 chars):

  ```
  जेवीएम का "कचरा संग्रह" लॉग निम्नानुसार स्वरूपित है:
  1. प्रत्येक पंक्ति टाइमस्टैम्प से शुरू होने वाले एकल कचरा संग्रह संचालन का प्रतिनिधित्व करती है।
  2. जेवीएम में विभिन्न मेमोरी क्षेत्रों के पहले और बाद के आकार को "मेमोरी एरिया: बिफोरसाइज-> आफ्टरसाइज (आवंटित आकार)" के रूप में दिखाया गया है, जहां मेमोरीएरिया PSYoungGen, ParOldGen, या मेटास्पेस में से एक है।
  3. यदि "MemoryArea:" को छोड़ दिया जाता है, तो यह संपूर्ण JVM की मेमोरी के पहले और बाद के आकार का प्रतिनिधित्व करता है।
  4. प्रत्येक पंक्ति मे
  ```

- **Generated length**: A100 = 3152 tokens, H100 = 3041 tokens  (Δ = +111)
- **First divergent token index**: 65  (of min(a,h) = 3041)
- **Matching tokens at aligned positions**: 91/3041 (3.0%)
- **Mean top-20 overlap** at aligned positions: 1.4/20
- **Token context around first divergence (idx=65)**:

  | pos | A100 tok | H100 tok |
  |---:|---|---|
  | 61 | `1667` | `1667` |
  | 62 | `279` | `279` |
  | 63 | `49272` | `49272` |
  | 64 | `58548` | `58548` |
  | 65 | `320` | `3118` ← diverge |
  | 66 | `68` | `389` |
  | 67 | `1301` | `4938` |
  | 68 | `306` | `5671` |
  | 69 | `553` | `1075` |

### Sample idx = 692
- **FP persistence**: A100→H100 45/49, H100→A100 49/49
- **Language**: hi
- **Prompt** (first 500 chars):

  ```
  मुझे नीचे दी गई कहानी का एक पंक्ति का सारांश दें:
  
  बहुत समय पहले उड़ीसा की रानी ने सुना था कि एक प्रस्तर मूर्तिकार था, जिसने देवताओं की ऐसी अद्भुत मूर्तियाँ बनाई थीं कि लोग मूर्तियों की सुंदरता देखकर खुशी से रो पड़ेंगे। एक गर्मी के दिन, शाही महल के मुख्य हॉल में राजा के साथ अकर्मण्यता से आराम करते हुए, रानी को अचानक एक विचार आया। "यह कितना अच्छा होगा, राजा, अगर हम भगवान जगन्नाथ की कुछ सुंदर मूर्तियों का निर्माण कर सकें ताकि लोग जगन्नाथ, बलभद्र और सुभद्रा की पूजा कर सकें। आप क्या सोचते हैं?"
  
  राज
  ```

- **Generated length**: A100 = 227 tokens, H100 = 186 tokens  (Δ = +41)
- **First divergent token index**: 137  (of min(a,h) = 186)
- **Matching tokens at aligned positions**: 137/186 (73.7%)
- **Mean top-20 overlap** at aligned positions: 15.2/20
- **Token context around first divergence (idx=137)**:

  | pos | A100 tok | H100 tok |
  |---:|---|---|
  | 133 | `230` | `230` |
  | 134 | `11` | `11` |
  | 135 | `14925` | `14925` |
  | 136 | `250` | `250` |
  | 137 | `42311` | `54575` ← diverge |
  | 138 | `116` | `47809` |
  | 139 | `34370` | `54784` |
  | 140 | `91217` | `113` |
  | 141 | `12619` | `91811` |

## Other recurring FPs (flagged in ≥5 configs, either direction)

| idx | A→H | H→A | A100 seq_len | H100 seq_len | lang | A/H gen-len | first divergence | match ratio | prompt |
|---:|---:|---:|---:|---:|:---:|---|---:|---:|---|
| 23 | 29 | 17 | 33 | 33 | en | 33/33 | None | 100% | Transcribe the recording into text. |
| 1009 | 6 | 29 | 83 | 83 | en | 83/83 | None | 100% | Compare the two given products and provide the advantages and disadvantages. |
| 534 | 11 | 11 | 7 | 7 | ch | 7/7 | None | 100% | 写一个五个词的句子来描述你的一天。 |
| 823 | 11 | 11 | 6 | 6 | ar | 6/6 | None | 100% | استخدم كلمة واحدة لوصف الطقس اليوم. |
| 97 | 20 | 1 | 128 | 116 | en | 128/116 | 3 | 3% | Compute the area of a rectangle with length 10cm and width 5cm. |
| 149 | 0 | 17 | 54 | 54 | en | 54/54 | 46 | 93% | Calculate the square root of a given number. |
| 144 | 0 | 14 | 489 | 320 | en | 489/320 | 28 | 9% | Find the cutoff score for successful completion of the exam |
| 55 | 0 | 14 | 179 | 176 | en | 179/176 | 12 | 8% | What is the force on a 1 kg mass due to the gravitational force? |
| 585 | 0 | 14 | 70 | 74 | ch | 70/74 | 26 | 37% | 为给定的产品创建一个口号。 |
| 36 | 3 | 9 | 100 | 79 | en | 100/79 | 74 | 94% | Analyze the given text for its tone. |
| 134 | 0 | 11 | 9 | 7 | en | 9/7 | 3 | 43% | Compose a five word sentence describing your day. |
| 72 | 0 | 10 | 62 | 73 | en | 62/73 | 20 | 69% | Describe the following person |
| 226 | 3 | 7 | 154 | 163 | sp | 154/163 | 15 | 20% | Encuentra la quinta potencia de -2. |
| 113 | 10 | 0 | 8 | 8 | en | 8/8 | None | 100% | Rewrite this sentence using the third person point of view. |
| 126 | 9 | 0 | 74 | 95 | en | 74/95 | 0 | 3% | Calculate the area of the triangle. |
| 11 | 1 | 7 | 8 | 8 | en | 8/8 | None | 100% | What is the capital of France? |
| 503 | 8 | 0 | 8 | 9 | ch | 8/9 | 1 | 12% | 生成一个 8 个字符的密码。 |
| 98 | 5 | 1 | 8 | 8 | en | 8/8 | None | 100% | Find the capital of Spain. |
| 441 | 5 | 0 | 80 | 80 | ch | 80/80 | None | 100% | 将给定的方程式转换为代数表达式。 |

## Notes

- Top-10 FP languages: en=6, hi=2, ch=1, ar=1
- A high *first divergence* index close to total length means the two runs agreed on most tokens and diverged late; a low index means the runs produced almost entirely different sequences.
- Samples where A100 and H100 produced *different generated sequences* inflate TOPLOC's per-position top-k mismatch even though neither model is cheating — this is a mode-switch, not a fraud signal. Any practical threshold has to allow for these.
