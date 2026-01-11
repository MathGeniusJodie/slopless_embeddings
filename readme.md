# toy example of fast/small vector embedding running on the cpu

this will become a library in the future for use in the WIP slopless web search engine

slopless will mainly use traditional search techniques, but vector search would be great to enhance certain queries

# classifier models to try
* https://huggingface.co/MongoDB/mdbr-leaf-ir (en)

# translation models to try
use all-> en for searches (server memory is limited) and a more accurate per-language model for indexing
* https://huggingface.co/Helsinki-NLP/opus-mt-mul-en

# rerankers to try
* https://huggingface.co/cross-encoder/ms-marco-MiniLM-L2-v2
* https://huggingface.co/cross-encoder/ms-marco-MiniLM-L4-v2
* https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
* https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2

# ideas for non-english query detection
* https://github.com/quickwit-oss/whichlang/
* custom solution: get uniquely english and uniquely non-english words from a corpus, probably msmarco

# plan
Since most content and users are english, we will first translate all content to english, then embed and index it.
When a query comes in, we will translate it to english, embed it, and search the index. english only models are much
faster and smaller for the quality than multilingual models. this will also ensure cross-language search works.

# supported languages wishlist
- [ ] English
- [ ] French
- [ ] Chinese
- [ ] Spanish
- [ ] Arabic
- [ ] Portuguese
- [ ] Indonesian
- [ ] Malay
- [ ] Japanese
- [ ] Russian
- [ ] German
- [ ] Italian
- [ ] Persian
- [ ] Polish
- [ ] Hindi
- [ ] Vietnamese