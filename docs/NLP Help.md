# From Zero to Email-NLP Analyst: A Hands-On Guide to Text Analytics, Topic Modeling, and ML on Email (Enron + Bilingual Edition, 2026)

The quickest way to become effective at email topic modeling is to work in this order: data engineering first (most of the signal in email is lost to quoted replies, duplicates and boilerplate before any model runs), then transparent baselines (TF-IDF, distinctive-term analysis, NMF/LDA), and only then embedding-based BERTopic with LLM labeling. Every code block below runs on public data: the CMU Enron maildir, the SetFit/enron_spam dataset on Hugging Face, sklearn's 20 Newsgroups, and SetFit/amazon_reviews_multi_es for Spanish.

## TL;DR

- **Most of the work is cleaning.** Email is full of forwards, quoted chains and duplicates. Strip quotes and signatures, deduplicate (exact hashes first, then MinHash near-duplicates) and use sent-mail folders before modeling. If you skip this, your "topics" will mostly be reply boilerplate and legal disclaimers.
- **Use two models side by side.** NMF on TF-IDF is a fast, transparent baseline. BERTopic (sentence-transformers → UMAP → HDBSCAN → c-TF-IDF, then LLM labels) usually gives the most readable topics on short email text. Don't pick a model on coherence scores alone: Hoyle et al. (NeurIPS 2021) showed automated coherence can declare a winner when human judges don't. Use coherence to shortlist candidates, then read the topics and run a word-intrusion check.
- **Pin versions in 2026.** Use Python 3.11/3.12 with gensim ≥4.4 (the first release with NumPy 2 support) and pandas 3.x, but keep pyLDAvis in an environment with `pandas<3` because it hasn't had a release since April 2023. Transformers v5 removed TensorFlow and renamed Trainer's `tokenizer=` to `processing_class=`. SetFit's v5 fixes are on its main branch but not yet in the 1.1.3 release. For bilingual Puerto Rico mail, use Lingua for language ID, `es_core_news_*` for Spanish, and `paraphrase-multilingual-MiniLM-L12-v2` (BERTopic's multilingual default) for embeddings.

---

## Key Findings

1. **The data sources are still live.** The Enron May 7, 2015 tarball (`enron_mail_20150507.tar.gz`, ~423 MB) is still served by CMU. EnronQA (arXiv 2505.00263) counts 517,401 emails across 150 users in this release, which CMU describes as "mostly senior management of Enron". It has no attachments. A Kaggle mirror (`wcukierski/enron-email-dataset`) packages the same version as one CSV.\[1\] For supervised work, `SetFit/enron_spam` has 33,716 labeled ham/spam emails with `text`, `label`, `label_text`, `subject` and `message` columns.\[2\] The original `amazon_reviews_multi` dataset is marked **defunct** on the Hub, but the Spanish port `SetFit/amazon_reviews_multi_es` (210k rows, 5 star classes) still loads.\[3\]\[4\]
2. **Duplication in Enron is extreme.** Each message can appear in the sender's `sent`, `sent_items`, `_sent_mail` and `all_documents` folders and in every recipient's inbox. One public EDA repo reports that 52.2% of parseable message bodies are byte-exact duplicates.\[5\] EnronQA (arXiv 2505.00263) confirms the scale independently: MinHash dedup cut the 2015 corpus from 517,401 to 228,098 documents, removing about 56%.
3. **Several tools in the brief are aging, so treat them carefully:**
   - **pyLDAvis** 3.4.1 (Apr 2023) predates NumPy 2 and pandas 3.\[6\]
   - **KeyBERT** 0.9.0 dates from Feb 2025.\[7\]
   - **talon** needs a `cchardet` shim on Python 3.11+.\[8\]
   - **GuidedLDA** is unmaintained.

   Use them when they help, but keep them in isolated environments or replace them with maintained equivalents (BERTopic's guided and zero-shot modes, NMF, scikit-learn).
4. **Automated coherence is a weak judge.** Coherence metrics (c_v, u_mass, NPMI) were validated for classical models. Hoyle et al. found that "automated evaluations declare a winning model when corresponding human evaluations do not."\[9\] Human review, word-intrusion checks and downstream usefulness are what actually count.
5. **LLM-native topic modeling works, but you pay per document.** TopicGPT (Pham et al., NAACL 2024) used GPT-4 to generate topics and GPT-3.5 to assign them. Its Table 11 reports "around $100 per dataset": $88 for Bills and $155 for the longer-document Wiki set. The authors also found that open-source Mistral-7B assigns topics well but generates them poorly. In practice the best trade-off for email is to cluster with BERTopic and have an LLM label only the ~50–200 clusters, not every email.

---

## Part 0 – Environment Setup (verified September 2026)

Current releases at the time of writing:

| Package | Version | Notes |
|---|---|---|
| pandas | 3.0.6 | 3.0 (Jan 2026) makes Copy-on-Write the only mode and infers a dedicated `str` dtype. Chained assignment no longer works. |\[10\]\[11\]
| scikit-learn | 1.9.1 | Includes `sklearn.cluster.HDBSCAN`. Its `min_samples` counts the point itself, so use one more than in standalone `hdbscan`. |\[12\]\[13\]\[14\]
| gensim | 4.4.0 | Oct 2025. Adds NumPy 2 support. 4.3.3 failed on SciPy ≥1.13 (`cannot import name 'triu'`). |\[15\]\[16\]
| spaCy | 3.8.x (3.8.16) | Models are v3.8.0. Python 3.13/3.14 wheels are available. |\[17\]
| BERTopic | 0.17.4 | Dec 2025. Dropped Python 3.9. Adds LiteLLM representation and a lightweight install option. |\[18\]\[19\]
| sentence-transformers | 6.0.x | `encode(sentences=...)` was renamed `encode(inputs=...)` in 5.4 (old name still warns). |\[20\]
| transformers | 5.x | PyTorch only. Trainer's `tokenizer=` became `processing_class=`, and `evaluation_strategy` became `eval_strategy`. |\[21\]
| setfit | 1.1.3 | Aug 2025. Transformers v5 fixes are merged on main but not released, so pin `transformers<5` in its environment. |\[22\]\[23\]
| umap-learn / hdbscan | 0.5.12 / 0.8.44 | |\[24\]\[25\]\[26\]
| keybert / yake | 0.9.0 / 0.7.3 | yake needs Python ≥3.10. |\[7\]\[27\]
| networkx | 3.7 | |\[28\]
| datasketch | 2.0.0 | New default MinHash scheme (`affine32`). Rebuild old indexes or pass `scheme="legacy"`. |\[29\]
| presidio-analyzer | 2.2.364 | Python 3.10–3.14. |\[30\]
| nltk | 3.10.3 | |\[31\]
| mail-parser | 4.6.5 | Prefers `extract-msg` for Outlook `.msg` files. |\[32\]\[33\]
| pyLDAvis | 3.4.1 | Last release Apr 2023. Isolate it with `pandas<3`. |\[6\]\[34\]\[35\]

Use **two environments** so the aging packages don't block the rest:

```bash
# Env 1: main analysis (Python 3.12)
python -m venv .venv-nlp && source .venv-nlp/bin/activate
pip install "pandas>=3.0,<3.1" "numpy>=2,<3" "scikit-learn>=1.9,<2.0" \
  "gensim>=4.4,<5" "spacy>=3.8,<3.9" "nltk>=3.10" "bertopic==0.17.4" \
  "sentence-transformers>=5.4" "transformers>=5,<6" "datasets>=4" \
  "umap-learn==0.5.12" "hdbscan==0.8.44" keybert==0.9.0 yake==0.7.3 \
  "networkx>=3.6" "datasketch>=2.0" presidio-analyzer presidio-anonymizer \
  lingua-language-detector beautifulsoup4 lxml matplotlib plotly litellm
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_lg   # Presidio default English model
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"

# Env 2: legacy/visual + SetFit (Python 3.11)
python -m venv .venv-legacy && source .venv-legacy/bin/activate
pip install "pandas<3" "numpy<2.3" "gensim>=4.4" pyLDAvis==3.4.1 \
  "setfit==1.1.3" "transformers<5" "sentence-transformers>=5,<6" "datasets>=4"
```

**Common pitfalls:**
- Colab ships a preinstalled NumPy, so a gensim built against an older NumPy fails at import. Use gensim 4.4+.
- Under pandas 3, `df[df.x > 0]["y"] = ...` silently does nothing (chained assignment). Use `df.loc[mask, "y"] = ...`.
- Code that checks `dtype == object` for text columns breaks under pandas 3. Use `pd.api.types.is_string_dtype`.
- Rebuild pickled datasketch LSH indexes after upgrading to 2.0.
- Old `Trainer(tokenizer=...)` tutorials throw errors under transformers v5.

---

## Part 1 – Email Data Engineering

### 1.1 Download and parse the Enron maildir

```bash
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
tar -xzf enron_mail_20150507.tar.gz      # produces ./maildir/<custodian>/<folder>/<n>.
```

The loader below walks the maildir and parses each file with Python's standard `email` package. It uses `policy.default`, which returns decoded, typed header objects. It keeps only **sent folders** by default, which removes most inbox duplication. The headers it extracts (From/To/Cc/Date/Subject/Message-ID/In-Reply-To) feed every later part of the guide.

```python
import os, email, hashlib
from email import policy
from email.utils import getaddresses, parsedate_to_datetime
import pandas as pd

SENT_FOLDERS = {"sent", "sent_items", "_sent_mail"}

def body_text(msg):
    """Return best-effort plain text; fall back to HTML->text; tolerate bad charsets."""
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    try:
        txt = part.get_content()
    except (LookupError, UnicodeDecodeError, KeyError):
        raw = part.get_payload(decode=True) or b""
        txt = raw.decode("latin-1", errors="replace")
    if part.get_content_type() == "text/html":
        from bs4 import BeautifulSoup
        txt = BeautifulSoup(txt, "lxml").get_text("\n")
    return txt

def addrs(msg, field):
    return [a.lower() for _, a in getaddresses(msg.get_all(field, [])) if a]

def load_enron(root="maildir", sent_only=True, limit=None):
    rows = []
    for custodian in sorted(os.listdir(root)):
        for dirpath, _, files in os.walk(os.path.join(root, custodian)):
            folder = os.path.relpath(dirpath, os.path.join(root, custodian)).split(os.sep)[0]
            if sent_only and folder not in SENT_FOLDERS:
                continue
            for fn in files:
                path = os.path.join(dirpath, fn)
                with open(path, "rb") as f:
                    msg = email.message_from_binary_file(f, policy=policy.default)
                try:
                    dt = parsedate_to_datetime(msg["Date"])
                except (TypeError, ValueError):
                    dt = None
                rows.append({
                    "path": path, "custodian": custodian, "folder": folder,
                    "message_id": str(msg["Message-ID"] or ""),
                    "in_reply_to": str(msg["In-Reply-To"] or ""),
                    "date": dt, "from": (addrs(msg, "From") or [""])[0],
                    "to": addrs(msg, "To"), "cc": addrs(msg, "Cc"),
                    "subject": str(msg["Subject"] or ""), "body": body_text(msg),
                })
                if limit and len(rows) >= limit:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)

df = load_enron(sent_only=True)
df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
df = df[(df.date >= "1999-01-01") & (df.date <= "2002-12-31")]   # drop bogus 1979/2044 dates
print(df.shape); df.head()
```

**Interpretation notes:**
- Enron headers include `X-From`/`X-To` display names and `X-Folder`. These are useful for mapping addresses to people.
- Enron has almost no `In-Reply-To` headers, which matters for threading (section 1.5).
- A few dates are wildly wrong, so always apply a date range filter.

**Try this:** run with `sent_only=False` and compare the row counts. The gap shows how much duplication the folder filter removes.

### 1.2 Strip quoted replies, forwards, signatures and disclaimers

Topic models will happily learn the topic "original message / forwarded by / please respond to". Remove this material *before* tokenizing. Start with regex heuristics tuned to Outlook/Lotus Notes (both common in Enron and in bank and insurer mailboxes), then use `talon` as a second pass.

```python
import re

CUT_PATTERNS = [
    r"^-{2,}\s*Original Message\s*-{2,}",                 # Outlook
    r"^-{5,}\s*Forwarded by",                              # Lotus Notes forward
    r"^\s*From:\s.+\n\s*(Sent|Date):\s",                   # inline header block
    r"^.+ on \d{2}/\d{2}/\d{2,4} \d{1,2}:\d{2}(:\d{2})? ?(AM|PM)?",  # Notes "X on 05/03/2001 10:12 AM"
    r"^On .+wrote:$",                                      # Gmail/Apple
    r"^El .+escribió:$",                                   # Spanish clients
    r"^-{2,}\s*Mensaje original\s*-{2,}",                  # Spanish Outlook
    r"^De:\s.+\n\s*(Enviado|Fecha):\s",
]
CUT_RE = re.compile("|".join(CUT_PATTERNS), re.MULTILINE | re.IGNORECASE)
DISCLAIMER_RE = re.compile(
    r"(this (e-?mail|message) (and any attachments )?(is|may be) (confidential|privileged).*|"
    r"este mensaje (es|puede ser) confidencial.*|\*{5,}.*)", re.IGNORECASE | re.DOTALL)
SIG_RE = re.compile(r"\n(--\s*\n|Thanks,?\s*\n|Regards,?\s*\n|Saludos,?\s*\n).{0,300}$",
                    re.IGNORECASE | re.DOTALL)

def strip_email(body: str) -> str:
    m = CUT_RE.search(body)
    new = body[: m.start()] if m else body
    new = "\n".join(l for l in new.splitlines() if not l.lstrip().startswith(">"))
    new = DISCLAIMER_RE.sub("", new)
    new = SIG_RE.sub("\n", new)
    return re.sub(r"\n{3,}", "\n\n", new).strip()

df["clean"] = df["body"].map(strip_email)
df["n_words"] = df["clean"].str.split().str.len()
print(df["n_words"].describe())
df = df[df.n_words >= 5]   # empty after stripping = pure forwards; analyze separately
```

**Library options:**
- **talon** (Mailgun) uses `quotations.extract_from_plain` / `extract_from_html`. Its `talon.signature.bruteforce.extract_signature` needs no ML. On Python 3.11+, insert `sys.modules['cchardet'] = chardet` before importing it.
- **email-reply-parser** (Zapier's port of GitHub's parser) is plain-text only and fast.
- **mail-parser** handles raw EML/MSG parsing, including Outlook `.msg` via `extract-msg`, which is useful for client exports.

```python
import sys, chardet; sys.modules["cchardet"] = chardet   # Py3.11+ shim
from talon import quotations
from talon.signature.bruteforce import extract_signature
def talon_clean(b):
    reply = quotations.extract_from_plain(b)
    text, _sig = extract_signature(reply)
    return text
df["clean_talon"] = df["body"].head(2000).map(talon_clean)
```

**Pitfall:** stripping forwards removes the forwarded content itself. For "what documents circulate" questions you may want to keep forwarded bodies as separate records. Decide per research question.

### 1.3 Deduplication: exact, then near-duplicate with MinHash

```python
from datasketch import MinHash, MinHashLSH

def norm(t): return re.sub(r"\W+", " ", t.lower()).strip()
df["hash"] = df["clean"].map(lambda t: hashlib.md5(norm(t).encode()).hexdigest())
df = df.sort_values("date").drop_duplicates("hash")          # exact dups

def shingles(t, k=5):
    w = norm(t).split()
    return {" ".join(w[i:i+k]) for i in range(max(1, len(w)-k+1))}

lsh = MinHashLSH(threshold=0.85, num_perm=128)
keep = []
for idx, text in zip(df.index, df["clean"]):
    m = MinHash(num_perm=128); m.update_batch([s.encode() for s in shingles(text)])
    if not lsh.query(m):          # no near-duplicate already kept
        lsh.insert(str(idx), m); keep.append(idx)
print(f"{len(df)} -> {len(keep)} after near-dup removal")
df = df.loc[keep]
```

**How it works:** MinHash estimates Jaccard similarity between shingle sets. LSH buckets the signatures so you avoid O(n²) comparisons.\[36\] A threshold of 0.85 on 5-word shingles catches templated mass emails and re-sent drafts. Lower it to 0.6–0.7 for "same announcement, lightly edited". SimHash is an alternative for very large corpora, but MinHash+LSH is the most commonly used approach in Python and is well documented.

### 1.4 Thread reconstruction

Use `In-Reply-To`/`References` when they exist, as they do in modern Exchange/M365 exports. Fall back to normalized subject plus participant overlap plus a time window, which is what you need for Enron.

```python
PREFIX = re.compile(r"^\s*((re|fw|fwd|rv|res)\s*:\s*)+", re.IGNORECASE)   # rv/res = Spanish
df["subj_norm"] = df["subject"].str.replace(PREFIX, "", regex=True).str.strip().str.lower()
df["participants"] = df.apply(lambda r: frozenset([r["from"], *r["to"], *r["cc"]]), axis=1)

df = df.sort_values("date")
thread_id, last_seen = {}, {}
for idx, r in df.iterrows():
    key = r["subj_norm"]
    prev = last_seen.get(key)
    if key and prev and (r.date - prev[1]).days <= 14 and (r.participants & prev[2]):
        thread_id[idx] = prev[0]
    else:
        thread_id[idx] = idx
    last_seen[key] = (thread_id[idx], r.date, r.participants)
df["thread_id"] = pd.Series(thread_id)
print(df.groupby("thread_id").size().describe())
```

**Try this:** model topics at the *thread* level by concatenating the cleaned bodies in a thread. Threads are longer than single emails, which directly addresses the short-text problem for LDA.

### 1.5 PII redaction with Presidio (English + Spanish)

On bank and insurance mailboxes, redact PII **before** any text leaves your VPC, including text sent to LLM labeling.

```python
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

provider = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_lg"},
               {"lang_code": "es", "model_name": "es_core_news_sm"}]})
analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en", "es"])
anonymizer = AnonymizerEngine()

def redact(text, lang="en"):
    res = analyzer.analyze(text=text, language=lang,
                           entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN",
                                     "CREDIT_CARD", "IBAN_CODE", "LOCATION"])
    return anonymizer.anonymize(text=text, analyzer_results=res).text

print(redact("Call Jeff Skilling at 713-853-6161 or jeff.skilling@enron.com"))
print(redact("Mi nombre es María Rivera, mi teléfono es 787-555-1234", lang="es"))
```

**Caveats:**
- Presidio's default recognizers are English-centric, and Spanish needs context words added (e.g. `correo`, `teléfono`).\[37\]
- Policy and claim numbers need custom `PatternRecognizer`s.
- Redacting names degrades NER and network analysis, so **pseudonymize** consistently (hash each name to a stable token) rather than deleting it.

---

## Part 2 – NLP Foundations for Analysis

### 2.1 Normalization, tokenization, lemmas, POS with spaCy

```python
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser"])  # keep tagger/lemmatizer/NER
EMAIL_STOP = {"enron", "ect", "hou", "cc", "subject", "pm", "am", "thanks", "please",
              "fyi", "com", "www", "http", "let", "know", "would", "could", "get", "also"}
STOP = nlp.Defaults.stop_words | EMAIL_STOP

def to_tokens(doc):
    return [t.lemma_.lower() for t in doc
            if t.is_alpha and len(t) > 2 and t.lemma_.lower() not in STOP
            and t.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}]

texts = df["clean"].tolist()
df["tokens"] = [to_tokens(d) for d in nlp.pipe(texts, batch_size=500, n_process=2)]
```

**Concepts and trade-offs:**
- **Tokenization** splits text into units. spaCy's tokenizer is rule-based and non-destructive.
- **Stemming** (Porter/Snowball) chops suffixes: fast, crude, and it produces non-words like "compani".
- **Lemmatization** maps words to dictionary forms ("meetings" → "meeting") using POS. It is better for interpretable topics.
- Filtering by POS to nouns, proper nouns, verbs and adjectives is the single most effective cleanup for LDA and NMF on email.
- **Domain stopwords** are the words that are frequent everywhere in *your* corpus (e.g. "ect", "hou" are Enron routing tokens). Find them empirically with the document-frequency check below.

```python
from collections import Counter
dfreq = Counter(w for toks in df.tokens for w in set(toks))
print([w for w, c in dfreq.most_common(60)])   # add routing/boilerplate terms to EMAIL_STOP, rerun
```

### 2.2 Phrases and collocations

```python
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
bigram = Phrases(df.tokens, min_count=20, threshold=10, connector_words=ENGLISH_CONNECTOR_WORDS)
trigram = Phrases(bigram[df.tokens], min_count=10, threshold=10)
df["tokens_ph"] = [trigram[bigram[t]] for t in df.tokens]
print(sorted({w for t in df.tokens_ph for w in t if "_" in w})[:50])  # e.g. natural_gas, power_plant
```

`Phrases` scores word pairs by how much more often they co-occur than chance would predict. Raise `threshold` to get fewer, stronger phrases. Phrases like `california_iso` and `conference_call` make topics far more readable.

### 2.3 Bag-of-words and TF-IDF: what the weights mean

```python
from sklearn.feature_extraction.text import TfidfVectorizer
docs = df.tokens_ph.str.join(" ")
tfidf = TfidfVectorizer(min_df=5, max_df=0.5, sublinear_tf=True)
X = tfidf.fit_transform(docs)
vocab = tfidf.get_feature_names_out()
row = X[0].toarray().ravel()
print(sorted(zip(row, vocab), reverse=True)[:10])
```

TF-IDF weight = (1 + log tf) × idf, with idf = log((1+N)/(1+df)) + 1, and each row is L2-normalized. A high weight means *frequent in this email and rare in the corpus*. That makes TF-IDF a relevance measure, not an importance measure. `max_df=0.5` drops words that appear in more than half of the emails, which is automatic domain-stopword removal. `min_df=5` drops typos and names that appear only once.

### 2.4 Distinctive terms across groups or time: log-odds with an informative Dirichlet prior

Raw frequency differences are dominated by common words, and plain log-ratios are dominated by rare ones. Monroe, Colaresi & Quinn (2008) fix both problems with a z-scored log-odds ratio that shrinks estimates toward a prior taken from the whole corpus.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def log_odds_dirichlet(docs_a, docs_b, prior_scale=0.01, min_df=5):
    cv = CountVectorizer(min_df=min_df); cv.fit(list(docs_a) + list(docs_b))
    ya = np.asarray(cv.transform(docs_a).sum(0)).ravel()
    yb = np.asarray(cv.transform(docs_b).sum(0)).ravel()
    alpha = (ya + yb) * prior_scale + 1e-3
    na, nb, a0 = ya.sum(), yb.sum(), alpha.sum()
    delta = (np.log((ya + alpha) / (na + a0 - ya - alpha))
             - np.log((yb + alpha) / (nb + a0 - yb - alpha)))
    var = 1 / (ya + alpha) + 1 / (yb + alpha)
    z = delta / np.sqrt(var)
    return pd.Series(z, index=cv.get_feature_names_out()).sort_values()

pre  = docs[df.date <  "2001-08-01"]; post = docs[df.date >= "2001-08-01"]
z = log_odds_dirichlet(post, pre)
print("Distinctive AFTER Aug-2001:", z.tail(20).index.tolist())
print("Distinctive BEFORE:", z.head(20).index.tolist())
```

|z| > 1.96 is roughly "significant". Use this to compare any two groups: custodians, departments, complaint versus non-complaint emails, Spanish versus English senders. `scattertext` renders the same analysis as an interactive plot for stakeholders.

### 2.5 Keyword extraction: TF-IDF vs YAKE vs KeyBERT

```python
import yake
from keybert import KeyBERT
sample = df["clean"].iloc[10]
yk = yake.KeywordExtractor(lan="en", n=3, top=8)
print("YAKE:", [k for k, s in yk.extract_keywords(sample)])      # lower score = better
kw = KeyBERT(model="all-MiniLM-L6-v2")
print("KeyBERT:", kw.extract_keywords(sample, keyphrase_ngram_range=(1, 3),
                                     stop_words="english", use_mmr=True, diversity=0.5, top_n=8))
```

| Method | How it works | Pros | Cons |
|---|---|---|---|
| TF-IDF | Needs the whole corpus | Fast, explainable | Unigram bias; needs a corpus |
| YAKE | Single-document statistical features | Unsupervised, language-aware (`lan="es"`) | Noisy on very short emails |
| KeyBERT | Embedding similarity between phrase and document, with MMR for diversity | Semantic, multilingual with the right model | Slower; last release Feb 2025 |

### 2.6 Named entities

```python
nlp_ner = spacy.load("en_core_web_sm")
ents = []
for doc in nlp_ner.pipe(df["clean"].head(5000), batch_size=200):
    ents += [(e.text.strip(), e.label_) for e in doc.ents if e.label_ in {"ORG", "PERSON", "GPE", "MONEY"}]
ent_df = pd.DataFrame(ents, columns=["text", "label"])
print(ent_df.groupby("label")["text"].value_counts().groupby(level=0).head(10))
```

**Interpretation:**
- Entities show *who and what* each topic involves: counterparties, regulators (FERC, CPUC), places.
- Small spaCy models make systematic errors on email, such as tagging "Thanks" or all-caps routing tokens as ORG. Always review the top entities.
- For production, use `en_core_web_trf` or add an `EntityRuler` with client-specific names (products, branches, policy types).

### 2.7 Sentiment and tone: VADER vs transformers

```python
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
vader = SentimentIntensityAnalyzer()
df["vader"] = df["clean"].map(lambda t: vader.polarity_scores(t)["compound"])
roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                   truncation=True, max_length=512)
df.loc[df.index[:2000], "roberta"] = [r["label"] for r in roberta(df["clean"].head(2000).tolist(), batch_size=32)]
print(df.groupby(df.date.dt.to_period("M"))["vader"].mean().tail(18))
```

**VADER vs transformers:**
- VADER is lexicon-based: instant and transparent, but blind to sarcasm, negation scope in long text and business euphemism. Business email is mostly neutral, so VADER compresses toward 0.
- Transformer models handle context better but are trained on tweets or reviews, a domain mismatch for email.
- Use sentiment for *relative* comparisons (between topics, over time), never as absolute truth. For Spanish use `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multilingual).

### 2.8 Similarity: cosine on TF-IDF vs embeddings

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
emb_model = SentenceTransformer("all-MiniLM-L6-v2")
E = emb_model.encode(df["clean"].tolist(), batch_size=128, show_progress_bar=True, normalize_embeddings=True)
q = emb_model.encode(["gas pipeline capacity problems in California"], normalize_embeddings=True)
top = np.argsort(-(E @ q.T).ravel())[:5]
print(df["clean"].iloc[top].str[:150].tolist())
print("TF-IDF cos of first two:", cosine_similarity(X[0], X[1])[0, 0])
```

TF-IDF cosine measures *lexical* overlap. Embeddings measure *semantic* overlap, so "outage" matches "power went down". Save `E` to disk, because BERTopic, clustering and classifiers all reuse it.

### 2.9 Exploratory visualization

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
df.set_index("date").resample("W").size().plot(ax=ax[0], title="Weekly sent volume")
df["n_words"].clip(upper=500).hist(bins=50, ax=ax[1]); ax[1].set_title("Words per email (clipped)")
df["from"].value_counts().head(15).plot.barh(ax=ax[2], title="Top senders")
plt.tight_layout(); plt.show()
```

The weekly volume curve lets you see major events, such as the Oct–Dec 2001 collapse, before any model does. The length histogram tells you how many emails fall into short-text territory (under ~30 words).

---

## Part 3 – Topic Modeling in Depth

### 3.1 Intuition

- **LDA** is a generative model. Each document is a mixture of topics θ ~ Dirichlet(α), each topic is a distribution over words φ ~ Dirichlet(η), and every word is drawn by first picking a topic, then a word. Low α means "each email is about few topics". Low η means "each topic uses few words".
- **NMF** factorizes the TF-IDF matrix as X ≈ W·H with non-negativity constraints. W is document-topic, H is topic-word. It is deterministic given a seed, fast, and often sharper on short text.
- **BERTopic** embeds documents, reduces dimensions (UMAP), clusters (HDBSCAN) and then describes each cluster with class-based TF-IDF. Each email gets *one* topic, unlike LDA's mixtures.

### 3.2 gensim LDA with coherence sweep

```python
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, CoherenceModel
dictionary = Dictionary(df.tokens_ph)
dictionary.filter_extremes(no_below=10, no_above=0.4, keep_n=20000)
corpus = [dictionary.doc2bow(t) for t in df.tokens_ph]

results = []
for k in [10, 15, 20, 30, 40]:
    lda = LdaMulticore(corpus, id2word=dictionary, num_topics=k, passes=10, iterations=200,
                       alpha="asymmetric", eta="auto", chunksize=2000, random_state=42, workers=3)
    cv = CoherenceModel(model=lda, texts=df.tokens_ph, dictionary=dictionary, coherence="c_v").get_coherence()
    npmi = CoherenceModel(model=lda, texts=df.tokens_ph, dictionary=dictionary, coherence="c_npmi").get_coherence()
    um = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence="u_mass").get_coherence()
    results.append((k, cv, npmi, um)); print(k, round(cv, 3), round(npmi, 3), round(um, 3))
```

**Hyperparameters:**
- `passes` is the number of full sweeps over the corpus. Keep increasing it until topics stop changing.
- `alpha="asymmetric"` lets some topics be globally common, which absorbs the residual boilerplate topic.
- `eta="auto"` learns word sparsity from the data.
- LdaMulticore does not support `alpha="auto"`. Use `LdaModel` if you need it.

**Coherence metrics:**
- **c_v** (sliding window, NPMI plus cosine) scored highest against human ratings in Röder, Both & Hinneburg (WSDM 2015), who searched about 237,912 combinations of coherence components. It is still known to reward odd word sets.
- **NPMI** is in [−1, 1]; higher is better.
- **u_mass** uses document co-occurrence in the training corpus and is ≤0, closer to 0 being better. It is cheap but weak.

Choose K where coherence *plateaus* and the topics are readable. Don't just take the argmax.

### 3.3 sklearn NMF and LDA

```python
from sklearn.decomposition import NMF, LatentDirichletAllocation
def top_words(H, vocab, n=10): return [[vocab[i] for i in row.argsort()[::-1][:n]] for row in H]

nmf = NMF(n_components=20, init="nndsvda", beta_loss="kullback-leibler", solver="mu",
          max_iter=400, random_state=42)
W = nmf.fit_transform(X)
for i, ws in enumerate(top_words(nmf.components_, vocab)): print(f"NMF {i}: {' '.join(ws)}")

cvec = CountVectorizer(min_df=5, max_df=0.5); C = cvec.fit_transform(docs)
sk_lda = LatentDirichletAllocation(n_components=20, learning_method="online", random_state=42,
                                   doc_topic_prior=0.1, topic_word_prior=0.01)
theta = sk_lda.fit_transform(C)
for i, ws in enumerate(top_words(sk_lda.components_, cvec.get_feature_names_out())): print(f"LDA {i}: {' '.join(ws)}")
```

LDA needs raw **counts** (CountVectorizer), because its likelihood is defined over counts. NMF works well on TF-IDF. The KL loss with the `mu` solver often gives more interpretable NMF topics on text than Frobenius loss.

### 3.4 pyLDAvis (legacy env)

```python
import pyLDAvis, pyLDAvis.gensim_models as gm
vis = gm.prepare(lda, corpus, dictionary, sort_topics=False)   # keep gensim's topic ids!
pyLDAvis.save_html(vis, "lda_vis.html")
```

**How to read it:**
- By default pyLDAvis re-sorts topics by size, so its "Topic 1" is not gensim's topic 0.\[38\] Always pass `sort_topics=False` so the visualization matches your tables.
- The λ slider: Sievert & Shirley's user study (29 users, a 50-topic model on 20 Newsgroups) "suggested that setting λ near 0.6 aids users in topic interpretation, although we expect this to vary across topics and data sets."
- Overlapping circles suggest topics you should merge.

### 3.5 Human evaluation: word intrusion

```python
import random
def intrusion_tasks(topics_words, n=5, seed=0):
    rnd = random.Random(seed); tasks = []
    for i, ws in enumerate(topics_words):
        other = rnd.choice([t for j, t in enumerate(topics_words) if j != i])
        intruder = next(w for w in other[:5] if w not in ws[:20])
        shown = ws[:n] + [intruder]; rnd.shuffle(shown)
        tasks.append({"topic": i, "words": shown, "answer": intruder})
    return pd.DataFrame(tasks)
tasks = intrusion_tasks(top_words(nmf.components_, vocab))
tasks.drop(columns="answer").to_csv("intrusion_for_annotators.csv", index=False)
```

Give the CSV to 2–3 domain experts. Each picks the word that doesn't belong in each row. The share of correct picks is the **model precision** (Chang et al., 2009, "Reading Tea Leaves"). Topics where the intruder is rarely found are incoherent regardless of their c_v score. An LLM can act as a cheap first-pass annotator, but calibrate it against at least one human.

### 3.6 Short-text issues specific to email

Many emails are shorter than 30 words ("ok, see you at 3"). LDA's per-document topic mixture is badly estimated from so few words. Mitigations:
- Aggregate to thread or sender-week documents.
- Drop emails under 5–10 content tokens, or model them separately.
- Prefer NMF or BERTopic, which assign one topic per document and so fit short text.
- Prepend the subject to the body.

### 3.7 Guided and seeded topic models

```python
from bertopic import BERTopic
seed_topic_list = [["gas", "pipeline", "capacity", "transport"],
                   ["california", "iso", "price", "cap", "ferc"],
                   ["meeting", "schedule", "conference", "call"],
                   ["contract", "legal", "agreement", "counterparty"]]
guided = BERTopic(seed_topic_list=seed_topic_list, min_topic_size=30)
zeroshot = BERTopic(embedding_model="all-MiniLM-L6-v2", min_topic_size=30,
                    zeroshot_topic_list=["Gas trading", "California energy crisis",
                                         "HR and recruiting", "Legal contracts"],
                    zeroshot_min_similarity=0.5)
```

- **Guided** mode nudges clusters toward seed words. **Zero-shot** mode assigns documents to named topics above a similarity threshold and clusters the rest freely. Zero-shot is ideal when a client already has a taxonomy (e.g. claims, billing, complaints) but you still want to discover unknown themes.
- **CorEx** (`corextopic`, anchor words) and **GuidedLDA** are the classical alternatives. GuidedLDA is unmaintained and hard to install on modern Python, so prefer BERTopic's modes or seeded NMF (initialize H with seed-word rows).

### 3.8 BERTopic end-to-end (the main event)

```python
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer

docs_bt = (df["subject"] + ". " + df["clean"]).str[:2000].tolist()
emb = SentenceTransformer("all-MiniLM-L6-v2").encode(docs_bt, batch_size=128, show_progress_bar=True)

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=40, min_samples=10, metric="euclidean",
                        cluster_selection_method="eom", prediction_data=True)
vectorizer = CountVectorizer(stop_words=list(STOP), min_df=5, ngram_range=(1, 2))
topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                       vectorizer_model=vectorizer,
                       ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                       representation_model={"KeyBERT": KeyBERTInspired(),
                                             "MMR": MaximalMarginalRelevance(diversity=0.3)},
                       calculate_probabilities=False, verbose=True)
topics, _ = topic_model.fit_transform(docs_bt, emb)
info = topic_model.get_topic_info(); print(info.head(25)[["Topic", "Count", "Name", "KeyBERT"]])
print("Outlier share:", (np.array(topics) == -1).mean())
```

**How to tune it:**
- `min_cluster_size` is the main granularity control.
- `n_neighbors` sets local vs global structure.
- `random_state` in UMAP makes the run reproducible.
- `reduce_frequent_words` down-weights ubiquitous terms in c-TF-IDF.
- Pre-computing `emb` lets you iterate quickly on everything downstream.

**Outliers, time, class, hierarchy:**

```python
new_topics = topic_model.reduce_outliers(docs_bt, topics, strategy="embeddings", embeddings=emb, threshold=0.3)
topic_model.update_topics(docs_bt, topics=new_topics, vectorizer_model=vectorizer)

tot = topic_model.topics_over_time(docs_bt, df["date"].dt.tz_localize(None).tolist(), nr_bins=36)
topic_model.visualize_topics_over_time(tot, top_n_topics=10).write_html("tot.html")

tpc = topic_model.topics_per_class(docs_bt, classes=df["custodian"].tolist())
topic_model.visualize_topics_per_class(tpc, top_n_topics=10).write_html("per_custodian.html")

hier = topic_model.hierarchical_topics(docs_bt)
topic_model.visualize_hierarchy(hierarchical_topics=hier).write_html("hierarchy.html")
topic_model.visualize_documents(docs_bt, embeddings=emb, sample=0.2).write_html("docs_map.html")
```

- HDBSCAN often labels 30–50% of emails as outliers (−1). BERTopic's author, Maarten Grootendorst, advises in GitHub Discussion #1738: "you would have to check yourself whether the new assignments make sense by inspecting a subset manually… it's actually advised not to reduce all of them." Keep a `-1` bucket in stakeholder reports as "unclassified / miscellaneous".
- **Hierarchy** is how you get from 80 fine topics to 10 executive themes.
- **Topics over time** (dynamic topic modeling) shows emergence and decay, for example California-crisis topics peaking in 2000–01.

**LLM labeling via LiteLLM (works with Bedrock, OpenAI, Anthropic and others):**

```python
import os
from bertopic.representation import LiteLLM
prompt = """I have a topic from corporate emails described by these keywords: [KEYWORDS]
Representative emails:
[DOCUMENTS]
Return a short business label (max 6 words) for this topic. Label:"""
llm_rep = LiteLLM(model="bedrock/anthropic.claude-3-haiku-20240307-v1:0", prompt=prompt,
                  nr_docs=4, diversity=0.1, doc_length=200, tokenizer="whitespace")
topic_model.update_topics(docs_bt, topics=new_topics, representation_model={"LLM": llm_rep, "KeyBERT": KeyBERTInspired()})
print(topic_model.get_topic_info()[["Topic", "Count", "LLM"]].head(20))
```

This sends only ~4 truncated, diverse documents per topic, so the cost scales with the number of topics, not emails. Swap in any LiteLLM model string (`gpt-4o-mini`, `ollama/llama3.1` for fully local). **Redact first** (section 1.5).

### 3.9 Top2Vec, TopicGPT and "cluster + summarize"

- **Top2Vec** jointly embeds documents and words and finds dense clusters. It is conceptually close to BERTopic but less modular, so choose BERTopic unless you need its word-vector search.
- **TopicGPT** prompts an LLM to generate a topic list with descriptions, refines it, then assigns every document with a quotable justification. It gives the best readability and user control, but costs LLM calls per document; the paper's Table 11 reports $88 for the Bills dataset and $155 for the longer-document Wiki dataset.
- **Cluster + summarize** (BERTopic + LLM labels, as above) is the pragmatic middle ground and the default recommendation.

### 3.10 Head-to-head comparison on the same emails

```python
def diversity(topic_words):  # share of unique words across all topics' top-10
    allw = [w for ws in topic_words for w in ws[:10]]; return len(set(allw)) / len(allw)

def npmi_of(topic_words):
    tw = [[w for w in ws if w in dictionary.token2id][:10] for ws in topic_words]
    tw = [t for t in tw if len(t) >= 3]
    return CoherenceModel(topics=tw, texts=df.tokens_ph, dictionary=dictionary, coherence="c_npmi").get_coherence()

bt_words = [[w for w, _ in topic_model.get_topic(t)] for t in info.Topic if t != -1]
bt_words = [[w.replace(" ", "_") for w in ws] for ws in bt_words]
cands = {"LDA-gensim": [[w for w, _ in lda.show_topic(i, 10)] for i in range(lda.num_topics)],
         "NMF": top_words(nmf.components_, vocab), "BERTopic": bt_words}
print(pd.DataFrame({m: {"NPMI": npmi_of(t), "diversity": diversity(t), "n_topics": len(t)}
                    for m, t in cands.items()}).T)
```

| Criterion | LDA | NMF | BERTopic |
|---|---|---|---|
| Short emails | Weak | Good | Best |
| Mixed-membership (one email, many topics) | Native | Via W weights | Approximate (`approximate_distribution`) |
| Speed on ~100k docs (CPU) | Minutes–hours | Minutes | Embedding-bound (GPU helps) |
| Determinism and explainability | Medium | High | Medium |
| Multilingual | Per-language pipelines | Per-language | One multilingual embedding |
| Stakeholder-ready labels | Manual | Manual | LLM labels built in |

**Recommendation:** run NMF as a sanity baseline and BERTopic as the primary model. Use LDA when you need proper mixed-membership probabilities, such as the share of an email's content about compliance. If NMF and BERTopic agree on a theme, it is robust. If they disagree, read the documents.

### 3.11 From topics to business insight

For each topic, build a one-row "topic card" containing:
- the label
- size and share of the corpus
- trend (slope over the last N months)
- top senders and departments
- mean sentiment
- key entities
- 3 representative emails

```python
df["topic"] = new_topics
cards = (df[df.topic != -1].groupby("topic")
         .agg(n=("clean", "size"), senders=("from", lambda s: s.value_counts().head(3).index.tolist()),
              sentiment=("vader", "mean"), first=("date", "min"), last=("date", "max")))
cards["label"] = topic_model.get_topic_info().set_index("Topic").loc[cards.index, "Name"]
cards["share"] = cards.n / cards.n.sum()
cards.sort_values("n", ascending=False).head(15).to_csv("topic_cards.csv")
```

Then answer stakeholder questions directly: which themes are growing, where negative tone concentrates, which teams own which themes, and what emerged recently. A theme like "claims-delay complaints up 40% quarter on quarter, mostly Spanish-language, concentrated in two branches" is an insight. A list of topic IDs is not.

---

## Part 4 – Communication and Network Analysis

```python
import networkx as nx
edges = (df.explode("to").dropna(subset=["to"])
           .query("to != '' and `from` != ''")
           .groupby(["from", "to"]).size().reset_index(name="w"))
G = nx.DiGraph(); G.add_weighted_edges_from(edges[["from", "to", "w"]].itertuples(index=False))
G = G.subgraph([n for n, d in G.degree() if d >= 3]).copy()

pr = nx.pagerank(G, weight="weight")
btw = nx.betweenness_centrality(G, k=min(500, len(G)), weight=None, seed=42)
UG = G.to_undirected()
comms = nx.community.louvain_communities(UG, weight="weight", seed=42)
node_comm = {n: i for i, c in enumerate(comms) for n in c}
cent = pd.DataFrame({"pagerank": pr, "betweenness": btw}).assign(community=pd.Series(node_comm))
print(cent.sort_values("pagerank", ascending=False).head(15))
print("modularity:", nx.community.modularity(UG, comms, weight="weight"))
```

**Reading the metrics:**
- **PageRank** measures who receives mail from important people.
- **Betweenness** identifies brokers who connect otherwise separate groups. These people are often the key informants in an investigation or process review.
- **Louvain communities** approximate the real organizational units.
- Only the ~150 custodians have complete mailboxes, so centrality is biased toward them. State this in any report.

**Combine network and topics** to see which community talks about what:

```python
df["comm"] = df["from"].map(node_comm)
ct = pd.crosstab(df.comm, df.topic, normalize="index")
print(ct.loc[:, ct.columns != -1].idxmax(axis=1).map(topic_model.get_topic_info().set_index("Topic")["Name"]))
```

**Response-time patterns** (these need inbox and sent mail, so reload with `sent_only=False` for this step):

```python
full = load_enron(sent_only=False, limit=200000)
full["date"] = pd.to_datetime(full["date"], utc=True, errors="coerce")
full["subj_norm"] = full["subject"].str.replace(PREFIX, "", regex=True).str.strip().str.lower()
full = full.drop_duplicates(["message_id"]).sort_values("date")
first = full.groupby("subj_norm").head(1)[["subj_norm", "from", "date"]].rename(columns={"from": "orig", "date": "t0"})
replies = full[full.subject.str.match(r"(?i)^\s*re:")].merge(first, on="subj_norm")
replies = replies[replies["from"] != replies["orig"]]
replies["hours"] = (replies.date - replies.t0).dt.total_seconds() / 3600
print(replies.query("0 < hours < 24*14").groupby(replies.date.dt.to_period("Q"))["hours"].median())
```

Median time to reply by quarter, by hour of day, or by topic works as a workload and escalation KPI on client mailboxes (for example, "billing complaints take 3× longer to get a first reply").

---

## Part 5 – Machine Learning on Text

### 5.1 Baselines on SetFit/enron_spam

```python
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

ds = load_dataset("SetFit/enron_spam")
tr, te = ds["train"].to_pandas(), ds["test"].to_pandas()
print(tr.label_text.value_counts())

clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=200_000),
                    LogisticRegression(C=4, max_iter=2000, class_weight="balanced"))
clf.fit(tr.text, tr.label); pred = clf.predict(te.text)
print(classification_report(te.label, pred, target_names=["ham", "spam"], digits=3))

svm = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True), LinearSVC(C=0.5))
svm.fit(tr.text, tr.label); print("LinearSVC acc:", (svm.predict(te.text) == te.label).mean())
```

- Expect roughly 98–99% here.\[39\] Spam is an easy task, so this dataset teaches the *workflow*, not how hard real routing problems are.
- For routing with many classes, report macro-F1 and a per-class confusion matrix.
- For imbalance, use `class_weight="balanced"`, stratified splits and threshold tuning on the precision-recall curve.

**Interpretability and error analysis:**

```python
vec, lr = clf.named_steps["tfidfvectorizer"], clf.named_steps["logisticregression"]
coefs = pd.Series(lr.coef_[0], index=vec.get_feature_names_out()).sort_values()
print("Ham-indicative:", coefs.head(20).index.tolist()); print("Spam-indicative:", coefs.tail(20).index.tolist())
te["proba"] = clf.predict_proba(te.text)[:, 1]
errors = te[(te.proba > 0.5) != (te.label == 1)].assign(conf=lambda d: (d.proba - 0.5).abs())
print(errors.sort_values("conf", ascending=False)[["label_text", "proba", "subject"]].head(15))
```

- Linear coefficients *are* the explanation: "enron", "ect" and "vince" as ham indicators show **leakage**, because the ham class is one company's mail. A model trained on this data learns "is this Enron?", not "is this spam?". That is exactly the kind of error analysis to do on client data too.
- For per-prediction explanations use LIME (`lime.lime_text.LimeTextExplainer`) or SHAP's `LinearExplainer` on the TF-IDF matrix.

### 5.2 Embeddings + classifier

```python
st = SentenceTransformer("all-MiniLM-L6-v2")
Etr = st.encode(tr.text.str[:2000].tolist(), batch_size=128, normalize_embeddings=True)
Ete = st.encode(te.text.str[:2000].tolist(), batch_size=128, normalize_embeddings=True)
elr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Etr, tr.label)
print(classification_report(te.label, elr.predict(Ete), digits=3))
```

Frozen embeddings plus a linear head is a strong, cheap baseline that transfers across languages if you swap in a multilingual encoder.

### 5.3 Few-shot with SetFit (legacy env, `transformers<5`)

```python
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
train_fs = sample_dataset(ds["train"], label_column="label", num_samples=16)   # 16 per class
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
args = TrainingArguments(batch_size=16, num_epochs=1, num_iterations=20)
trainer = Trainer(model=model, args=args, train_dataset=train_fs, eval_dataset=ds["test"].select(range(2000)),
                  column_mapping={"text": "text", "label": "label"})
trainer.train(); print(trainer.evaluate())
```

SetFit contrastively fine-tunes the encoder on pairs built from a handful of labels, then fits a logistic head. The SetFit authors (Tunstall et al., 2022) report that "with only 8 labeled examples per class on the Customer Reviews sentiment dataset, SetFit is competitive with fine-tuning RoBERTa Large on the full training set of 3k examples." That makes it ideal when a client can label only ~100 emails.

### 5.4 Zero-shot: NLI and LLMs

```python
zs = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
labels = ["billing or payment", "claim status", "complaint", "meeting scheduling", "legal or contract"]
print(zs("Todavía no he recibido el pago de mi reclamación del mes pasado.", candidate_labels=labels,
         hypothesis_template="This email is about {}.", multi_label=False))
```

- Always pass `model=` explicitly.
- NLI zero-shot runs one forward pass per label, so latency grows with the number of labels.
- LLM zero-shot (a Bedrock/OpenAI call with a JSON schema) handles nuanced taxonomies better but costs cents per email. Use it to **create silver labels** and distill them into SetFit or TF-IDF models for production.

### 5.5 Fine-tuning a small transformer (transformers v5 API)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments as TA, Trainer as HFTrainer, DataCollatorWithPadding
import evaluate
ckpt = "distilbert-base-multilingual-cased"   # or "answerdotai/ModernBERT-base" (English), "xlm-roberta-base"
tok = AutoTokenizer.from_pretrained(ckpt)
enc = ds.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True)
mdl = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
f1 = evaluate.load("f1")
def metrics(p): return f1.compute(predictions=p.predictions.argmax(-1), references=p.label_ids, average="macro")
args = TA(output_dir="ft-enron", eval_strategy="epoch", save_strategy="epoch", learning_rate=3e-5,
          per_device_train_batch_size=32, num_train_epochs=2, weight_decay=0.01, load_best_model_at_end=True,
          metric_for_best_model="f1", report_to="none")
trainer = HFTrainer(model=mdl, args=args, train_dataset=enc["train"].shuffle(seed=42).select(range(10000)),
                    eval_dataset=enc["test"], processing_class=tok,
                    data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
trainer.train()
```

v5 specifics: `processing_class=tok` (not `tokenizer=`), `eval_strategy` (not `evaluation_strategy`), and `report_to` now defaults to `"none"`.

### 5.6 Weak supervision: topics → labels → classifier

```python
topic_to_label = {0: "gas_trading", 3: "california_crisis", 7: "scheduling"}   # fill after reading topic cards
weak = df[df.topic.isin(topic_to_label)].assign(y=lambda d: d.topic.map(topic_to_label))
weak = weak[pd.Series(new_topics, index=df.index).loc[weak.index] == weak.topic]
wclf = make_pipeline(TfidfVectorizer(min_df=3, ngram_range=(1, 2)), LogisticRegression(max_iter=2000, class_weight="balanced"))
wclf.fit(weak.clean, weak.y)
```

Topic assignments are noisy labels. Train on the high-confidence core (non-outliers), hand-label about 200 emails as a gold test set, and measure the result. This is how an exploratory analysis becomes a production router.

### 5.7 Clustering for discovery

```python
from sklearn.cluster import KMeans, HDBSCAN as SkHDBSCAN
km = KMeans(n_clusters=30, n_init="auto", random_state=42).fit(E)
sk_hdb = SkHDBSCAN(min_cluster_size=40, min_samples=11).fit(UMAP(n_components=10, random_state=42).fit_transform(E))
print(pd.Series(sk_hdb.labels_).value_counts().head())
```

### 5.8 When classic methods beat LLMs, and vice versa

| Situation | Best choice |
|---|---|
| 100k+ emails per day, latency <10 ms, auditable | TF-IDF + linear model |
| Fewer than 100 labels, multilingual | SetFit on a multilingual encoder |
| Taxonomy changing weekly, no labels | NLI or LLM zero-shot, then distill |
| Nuanced intent or extraction (why is the customer angry?) | LLM with a structured-output schema |
| Exploratory themes | BERTopic + LLM labels on clusters only |

A linear model on CPU costs essentially nothing per email. LLM calls cost fractions of a cent to cents each and take hundreds of milliseconds. At mailbox scale, LLMs belong at the labeling and distillation step or on the long tail, not on every message.

---

## Part 5b – Spanish and Bilingual (Code-Switched) Email

```python
from lingua import Language, LanguageDetectorBuilder
detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.SPANISH).with_preloaded_language_models().build()

def lang_profile(text):
    conf = detector.compute_language_confidence_values(text)
    return {c.language.iso_code_639_1.name.lower(): round(c.value, 3) for c in conf}

print(lang_profile("Hola, te envío el claim form, please sign it antes del viernes."))
for r in detector.detect_multiple_languages_of("Buenos días, adjunto la póliza. Let me know if you need anything else."):
    print(r.language.name, r.start_index, r.end_index)
```

- Restricting Lingua to the languages you actually expect makes it faster and more accurate. Lingua is designed for short and mixed text, which is exactly what email is.\[40\]
- `detect_multiple_languages_of` segments code-switched text.
- fastText `lid.176` is faster for very high volume but weaker on short snippets.\[40\]

**Spanish pipeline on real data (`SetFit/amazon_reviews_multi_es`):**

```python
es = load_dataset("SetFit/amazon_reviews_multi_es", split="train").shuffle(seed=0).select(range(20000)).to_pandas()
nlp_es = spacy.load("es_core_news_sm", disable=["parser", "ner"])
from nltk.corpus import stopwords
ES_STOP = set(stopwords.words("spanish")) | nlp_es.Defaults.stop_words | {"producto", "amazon"}
def es_tokens(doc): return [t.lemma_.lower() for t in doc if t.is_alpha and len(t) > 2
                            and t.lemma_.lower() not in ES_STOP and t.pos_ in {"NOUN", "VERB", "ADJ"}]
es["tokens"] = [es_tokens(d) for d in nlp_es.pipe(es.text, batch_size=500)]
print(es[["text", "tokens"]].head(3))

z_es = log_odds_dirichlet(es.tokens[es.label == 0].str.join(" "), es.tokens[es.label == 4].str.join(" "))
print("1-star distinctive:", z_es.tail(15).index.tolist())
```

**Multilingual BERTopic** clusters English and Spanish emails about the same issue together:

```python
ml_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
es_emb = ml_model.encode(es.text.tolist(), batch_size=128, show_progress_bar=True)
es_vec = CountVectorizer(stop_words=list(ES_STOP | STOP), min_df=5)
es_topics = BERTopic(language="multilingual", vectorizer_model=es_vec, min_topic_size=40).fit(es.text.tolist(), es_emb)
print(es_topics.get_topic_info().head(15))
```

**Bilingual rules of thumb:**
- Route each email to a language-specific spaCy pipeline for lemmas and NER, but embed everything with one multilingual encoder.
- Put a **union of EN+ES stopwords** in the c-TF-IDF vectorizer.
- Keep Puerto Rico-specific vocabulary ("CRIM", "ASUME", municipality names) and anglicisms ("el claim", "el deductible") in the vocabulary, and don't translate them away.
- Spanish lemmatization with `sm` models is decent. Use `es_core_news_lg` or `es_dep_news_trf` for accuracy.
- Accents matter for meaning, so don't strip them before lemmatization. Stripping them for dedup hashing is fine.
- Lower-case the text but keep `ñ`.
- For LLM topic labels, ask for labels in the stakeholder's language.

---

## Part 6 – Capstone: End-to-End Enron Analysis

Run these in order. Each step reuses the objects defined above.

1. **Ingest:** `load_enron(sent_only=True)`, then filter dates.
2. **Clean:** `strip_email`, drop emails under 5 words, exact and MinHash dedup, `thread_id`.
3. **EDA:** weekly volume, length distribution, top senders, domain-stopword check.
4. **Linguistic features:** spaCy tokens → Phrases → TF-IDF. Log-odds of the pre- vs post-Aug-2001 vocabulary.
5. **Topics:** NMF (k=20) baseline, BERTopic primary, outlier reduction, LLM labels, hierarchy to ~10 themes, word-intrusion check on the top 20 topics.
6. **Entities and sentiment:** entities per topic, VADER plus transformer sentiment per topic per month.
7. **Network:** PageRank, betweenness, Louvain; community × topic crosstab.
8. **Time:** topics over time; response-time medians by quarter.
9. **Classification:** weakly supervised topic router, evaluated on 200 hand-labeled emails; spam/ham model as a filter.
10. **Deliverables:** `topic_cards.csv`, `tot.html`, `hierarchy.html`, `docs_map.html`, a network figure, and a 1-page memo covering the top 5 findings, each backed by numbers and 2 quoted emails.

```python
summary = {
  "emails_after_clean": len(df), "threads": df.thread_id.nunique(),
  "topics": int((topic_model.get_topic_info().Topic != -1).sum()),
  "outlier_share": float((np.array(new_topics) == -1).mean()),
  "top_growing_topics": tot.groupby("Topic").apply(lambda g: np.polyfit(range(len(g)), g.Frequency, 1)[0])
                           .sort_values(ascending=False).head(5).index.tolist(),
  "top_brokers": cent.sort_values("betweenness", ascending=False).head(5).index.tolist()}
print(summary)
```

**What to look for in Enron specifically:**
- Energy trading and California-market topics dominate 2000–01.
- Post-August 2001, the log-odds terms shift toward stock, employee and bankruptcy vocabulary.
- Betweenness picks out executive assistants and trading-desk leads as brokers.
- Tone moves negative in Q4 2001.

If your pipeline doesn't show these, suspect your cleaning first.

### Reusable playbook for bank and insurance mailboxes

1. **Governance:** get a data processing agreement, confirm the lawful basis and retention rules, and decide where processing happens (VPC or Bedrock).
2. **Redaction first:** run Presidio (EN+ES) with custom recognizers for policy, claim and account numbers, and pseudonymize names consistently.
3. **Export format:** PST/MSG/EML. Parse with `email` or `mail-parser` and keep Message-ID, In-Reply-To and References.
4. **Language ID per email and per segment** with Lingua. Record the language mix as a first-class variable.
5. **Strip content:** quotes, forwards and disclaimers with EN+ES patterns. Measure the share of emails emptied by stripping.
6. **Deduplicate:** exact, then MinHash. Choose one record per thread for topic modeling.
7. **Baselines:** volume over time, senders and queues, distinctive terms by queue and month.
8. **Topics:** NMF baseline plus multilingual BERTopic, LLM labels in the stakeholder's language, hierarchy for executives.
9. **Validate:** word intrusion with 2 SMEs, 50-email spot check per top topic, stability across 3 seeds.
10. **Enrich:** entities (products, branches), sentiment by topic, response-time KPIs.
11. **Operationalize:** convert topics to a taxonomy, silver-label with the LLM, hand-label a gold set, deploy SetFit or a linear router, monitor drift monthly.
12. **Report:** topic cards, trends, owners, recommended actions, and caveats (coverage, bias, outliers).

---

## Recommendations

- **For the upcoming project:** spend the first week on Part 1. Measure how much text quote-stripping and dedup remove, because that number predicts topic quality.
- **Model choice:** make BERTopic with a multilingual encoder and LLM labels the primary model, and keep NMF as the auditable baseline. Report where the two disagree.
- **Validation:** never ship a topic model on coherence alone. Budget 2–3 SME hours for word intrusion and document spot checks.
- **Isolation:** keep the pyLDAvis and SetFit environments separate until those packages release pandas-3 and transformers-v5 compatible versions.
- **Cost:** use LLMs at the cluster-labeling and silver-labeling stages, then distill to cheap models for per-email scoring.

## Caveats

- Version facts reflect PyPI and release notes as of late September 2026. SetFit's transformers-v5 support is merged on main but unreleased,\[23\] and pyLDAvis has no documented pandas-3 test, so verify these in your own environment.
- The 52.2% byte-exact duplicate figure comes from a single public repository. EnronQA's MinHash dedup (517,401 → 228,098 documents, about 56% removed) independently supports the scale, but the two measure slightly different things.
- The Enron spam dataset leaks the company identity into the ham class, so its accuracy numbers overstate real-world difficulty.
- The dataset and model names in the code were checked against the Hub at the time of writing. Hub repositories can be renamed or gated, as `amazon_reviews_multi` was.\[4\]

## Further Learning

**Books**
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft, free online).
- Alammar & Grootendorst, *Hands-On Large Language Models* (O'Reilly, 2024); its topic modeling chapter is BERTopic-centric.
- Bird, Klein & Loper, *Natural Language Processing with Python* (the NLTK book).
- Vajjala et al., *Practical Natural Language Processing*.

**Docs and courses**
- BERTopic docs, including the "Best Practices" page.
- spaCy course (course.spacy.io).
- Hugging Face LLM/NLP course.
- gensim tutorials; sbert.net.

**Papers**
- Blei, Ng & Jordan (2003), Latent Dirichlet Allocation.
- Lee & Seung (1999) on NMF.
- Grootendorst (2022), BERTopic: neural topic modeling with a class-based TF-IDF procedure.
- Chang et al. (2009), Reading Tea Leaves.
- Newman et al. (2010) and Lau et al. (2014) on coherence/NPMI.
- Röder et al. (2015), Exploring the space of topic coherence measures.
- Hoyle et al. (2021), Is Automated Topic Model Evaluation Broken?
- Monroe, Colaresi & Quinn (2008), Fightin' Words.
- Blei & Lafferty (2006), Dynamic Topic Models.
- Tunstall et al. (2022), SetFit.
- Yin et al. (2019), zero-shot classification via NLI.
- Pham et al. (2024), TopicGPT.
- Klimt & Yang (2004), the Enron corpus.

## Glossary

- **Bag-of-words (BoW):** a document represented as word counts, ignoring order.
- **c-TF-IDF:** TF-IDF computed per cluster (all of a cluster's documents joined) to find words that describe that cluster.
- **Coherence (c_v, u_mass, NPMI):** automated scores of how much a topic's top words co-occur.
- **Collocation / phrase:** a word sequence that co-occurs more often than chance (e.g. "natural_gas").
- **Code-switching:** mixing languages within a message or sentence.
- **Dirichlet prior (α, η):** controls the sparsity of document-topic and topic-word distributions in LDA.
- **Embedding:** a dense vector capturing meaning; similar texts get nearby vectors.
- **HDBSCAN:** density-based clustering that finds variable-density clusters and labels noise as −1.
- **Lemma / stem:** the dictionary form of a word vs a crude suffix-stripped form.
- **Log-odds with informative Dirichlet prior:** a z-scored measure of how distinctive a word is between two corpora.
- **MinHash / LSH:** hashing techniques that estimate Jaccard similarity and find near-duplicates without comparing every pair.
- **NER:** named entity recognition (people, organizations, places, money).
- **NLI zero-shot:** classifying by testing whether the text entails "This is about {label}".
- **NMF:** non-negative matrix factorization; X ≈ WH.
- **POS tagging:** labeling tokens with part of speech.
- **Stopwords:** high-frequency, low-information words; include corpus-specific ones.
- **TF-IDF:** term frequency × inverse document frequency.
- **UMAP:** non-linear dimensionality reduction that preserves local neighborhoods.
- **Weak supervision:** training on noisy, programmatically generated labels.
- **Word intrusion:** a human test where annotators spot an injected word among a topic's top words.

## Sources

1. [The Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)
2. [SetFit/enron\_spam · Datasets at Hugging Face](https://huggingface.co/datasets/SetFit/enron_spam)
3. [SetFit/amazon\_reviews\_multi\_es · Datasets at Hugging Face](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_es)
4. [defunct-datasets/amazon\_reviews\_multi · Datasets at Hugging Face](https://huggingface.co/datasets/defunct-datasets/amazon_reviews_multi)
5. [GitHub - Exios66/Enron-Evaluation-Environment: Exploratory data analysis of the CMU classic Enron email corpus, and the production of a pipeline-ready correspondence dataset for the llm-mailroom document-processing pipeline. · GitHub](https://github.com/Exios66/Enron-Evaluation-Environment)
6. [pyLDAvis · PyPI](https://translate.google.com/translate?u=https%3A%2F%2Fpypi.org%2Fproject%2FpyLDAvis%2F&hl=id&sl=en&tl=id&client=srp)
7. [keybert 0.9.0 on PyPI - Libraries.io - security & maintenance data for open source software](https://libraries.io/pypi/keybert)
8. [Email Reply Extraction with Talon](https://www.agentmail.to/docs/talon-reply-extraction)
9. [Is Automated Topic Model Evaluation Broken?: The Incoherence of Coherence](https://proceedings.neurips.cc/paper/2021/file/0f83556a305d789b1d71815e8ea4f4b0-Paper.pdf)
10. [pandas · PyPI](https://pypi.org/project/pandas/)
11. [Pandas 3.0 Released!](https://pandas.pydata.org/community/blog/pandas-3.0.html)
12. [scikit-learn · PyPI](https://pypi.org/project/scikit-learn/)
13. [sklearn.cluster — scikit-learn 1.9.1 documentation](https://scikit-learn.org/stable/api/sklearn.cluster.html)
14. [HDBSCAN — scikit-learn 1.9.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)
15. [Releases · piskvorky/gensim](https://github.com/piskvorky/gensim/releases)
16. [python-gensim - AUR (en) - Arch Linux](https://aur.archlinux.org/packages/python-gensim)
17. [spacy · PyPI](https://pypi.org/project/spacy/)
18. [Changelog - BERTopic](https://maartengr.github.io/BERTopic/changelog.html)
19. [MaartenGr/BERTopic: v0.17.4](https://zenodo.org/records/17803687)
20. [Migration Guide — Sentence Transformers documentation](https://sbert.net/docs/migration_guide.html)
21. [Hugging Face Transformers Tutorial (2026): Pipeline & Models](https://www.assemblyai.com/blog/hugging-face-transformers-tutorial)
22. [setfit · PyPI](https://pypi.org/project/setfit/)
23. [Fix default\_logdir import for transformers v5 by elias-aouad · Pull Request #634 · huggingface/setfit](https://github.com/huggingface/setfit/pull/634)
24. [umap-learn](https://pypi.org/project/umap-learn/)
25. [hdbscan · PyPI](https://pypi.org/project/hdbscan/0.8.4/)
26. [umap-learn 0.1.3](https://pypi.org/project/umap-learn/0.1.3/)
27. [yake · PyPI](https://pypi.org/project/yake/)
28. [networkx · PyPI](https://pypi.org/project/networkx/)
29. [datasketch - PyPI Package Security Analysis - Socket](https://socket.dev/pypi/package/datasketch)
30. [presidio-analyzer · PyPI](https://pypi.org/project/presidio-analyzer/)
31. [nltk · PyPI](https://pypi.org/project/nltk/)
32. [mail-parser](https://pypi.python.org/pypi/mail-parser)
33. [mail-parser - PyPI Package Security Analysis - Socket](https://socket.dev/pypi/package/mail-parser)
34. [pyLDAvis - Python Package Health Analysis](https://snyk.io/advisor/python/pyldavis)
35. [Releases · bmabey/pyLDAvis](https://github.com/bmabey/pyLDAvis/releases)
36. [MinHash for approximate deduplication](https://theneuralbase.com/chunking-strategies/learn/advanced/minhash-for-approximate-deduplication/)
37. [presidio/docs/analyzer/languages.md at main · microsoft/presidio](https://github.com/microsoft/presidio/blob/main/docs/analyzer/languages.md)
38. [Gensim topic modeling: refuses to run on pandas 2; pyLDAvis topic numbers don't match the csv; no per-document topics · Issue #1667 · NLP-Suite/NLP-Suite](https://github.com/NLP-Suite/NLP-Suite/issues/1667)
39. [mrm8488/bert-tiny-finetuned-enron-spam-detection · Hugging Face](https://huggingface.co/mrm8488/bert-tiny-finetuned-enron-spam-detection)
40. [An accurate natural language detection library, suitable for short text and mixed-language text](https://pypi.org/project/lingua-language-detector/1.3.5)
