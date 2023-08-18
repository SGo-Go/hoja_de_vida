****Module 1 Resources****
==========================

****

****

****NLP tasks****

-   [*Hugging Face tasks page*](https://huggingface.co/tasks)
-   [*Hugging Face NLP course chapter 7: Main NLP
    Tasks*](https://huggingface.co/course/chapter7/1?fw=pt)
-   Background reading on specific tasks

    -   Summarization: [*Hugging Face summarization task
        page*](https://huggingface.co/tasks/summarization) and [*course
        section*](https://huggingface.co/learn/nlp-course/chapter7/5)
    -   Sentiment Analysis: [*Blog on "Getting Started with Sentiment
        Analysis using
        Python"*](https://huggingface.co/blog/sentiment-analysis-python)
    -   Translation: [*Hugging Face translation task
        page*](https://huggingface.co/docs/transformers/tasks/translation)
        and [*course
        section*](https://huggingface.co/learn/nlp-course/chapter7/4)
    -   Zero-shot classification: [*Hugging Face zero-shot
        classification task
        page*](https://huggingface.co/tasks/zero-shot-classification)
    -   Few-shot learning: [*Blog on "Few-shot learning in practice:
        GPT-Neo and the 🤗 Accelerated Inference
        API"*](https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api)

[*****Hugging Face Hub*****](https://huggingface.co/docs/hub/index)

-   [*Models*](https://huggingface.co/models)
-   [*Datasets*](https://huggingface.co/datasets)
-   [*Spaces*](https://huggingface.co/spaces)

****Hugging Face libraries****

-   [*Transformers*](https://huggingface.co/docs/transformers/index)

    -   Blog post on inference configuration: [*How to generate text:
        using different decoding methods for language generation with
        Transformers*](https://huggingface.co/blog/how-to-generate)

-   [*Datasets*](https://huggingface.co/docs/datasets)
-   [*Evaluate*](https://huggingface.co/docs/evaluate/index)

****Models****

-   Base model versions of models used in the demo notebook

    -   [*T5*](https://huggingface.co/docs/transformers/model_doc/t5)
    -   [*BERT*](https://huggingface.co/docs/transformers/model_doc/bert)
    -   [*Marian NMT
        framework*](https://huggingface.co/docs/transformers/model_doc/marian)
        (with 1440 language translation models!)
    -   [*DeBERTa*](https://huggingface.co/docs/transformers/model_doc/deberta)
        (Also see
        [*DeBERTa-v2*](https://huggingface.co/docs/transformers/model_doc/deberta-v2))
    -   [*GPT-Neo*](https://huggingface.co/docs/transformers/model_doc/gpt_neo)
        (Also see
        [*GPT-NeoX*](https://huggingface.co/docs/transformers/model_doc/gpt_neox))

-   [*Table of
    LLMs*](https://crfm.stanford.edu/ecosystem-graphs/index.html)

****Prompt engineering****

-   [*Best practices for OpenAI-specific
    models*](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)
-   [*DAIR.AI guide*](https://www.promptingguide.ai/)
-   [*ChatGPT Prompt Engineering
    Course*](https://learn.deeplearning.ai/chatgpt-prompt-eng) by OpenAI
    and DeepLearning.AI
-   [*🧠 *](https://github.com/f/awesome-chatgpt-prompts)[*Awesome
    ChatGPT Prompts*](https://github.com/f/awesome-chatgpt-prompts) for
    fun examples with ChatGPT

Module 2 Resources
==================

Research papers on increasing context length limitation

-   [*Pope et al 2022*](https://arxiv.org/abs/2211.05102)
-   [*Fu et al 2023*](https://arxiv.org/abs/2212.14052)

Industry examples on using vector databases

-   FarFetch

    -   [*FarFetch: Powering AI With Vector Databases: A Benchmark -
        Part
        I*](https://www.farfetchtechblog.com/en/blog/post/powering-ai-with-vector-databases-a-benchmark-part-i/)
    -   [*FarFetch: Powering AI with Vector Databases: A Benchmark -
        Part
        2*](https://www.farfetchtechblog.com/en/blog/post/powering-ai-with-vector-databases-a-benchmark-part-ii/)
    -   [*FarFetch: Multimodal Search and Browsing in the FARFETCH
        Product Catalogue - A primer for conversational
        search*](https://www.farfetchtechblog.com/en/blog/post/multimodal-search-and-browsing-in-the-farfetch-product-catalogue-a-primer-for-conversational-search/)

-   [*Spotify: Introducing Natural Language Search for Podcast
    Episodes*](https://engineering.atspotify.com/2022/03/introducing-natural-language-search-for-podcast-episodes/) 
-   [*Vector Database Use Cases compiled by
    Qdrant*](https://qdrant.tech/use-cases/)

Vector indexing strategies 

-   Hierarchical Navigable Small Worlds (HNSW) 

    -   [*Malkov and Yashunin 2018*](https://arxiv.org/abs/1603.09320)

-   Facebook AI Similarity Search (FAISS)

    -   [*Meta AI Blog*](https://ai.facebook.com/tools/faiss/)

-   Product quantization

    -   [*PQ for Similarity Search by Peggy
        Chang*](https://towardsdatascience.com/product-quantization-for-similarity-search-2f1f67c5fddd) 

Cosine similarity and L2 Euclidean distance 

-   [*Cosine and L2 are functionally the same when applied on normalized
    embeddings*](https://stats.stackexchange.com/questions/146221/is-cosine-similarity-identical-to-l2-normalized-euclidean-distance)

Filtering methods

-   [*Filtering: The Missing WHERE Clause in Vector Search by
    Pinecone*](https://www.pinecone.io/learn/vector-search-filtering/)

Chunking strategies

-   [*Chunking Strategies for LLM applications by
    Pinecone*](https://www.pinecone.io/learn/chunking-strategies/)
-   [*Semantic Search with Multi-Vector Indexing by
    Vespa*](https://blog.vespa.ai/semantic-search-with-multi-vector-indexing/) 

Other general reading

-   [*Vector Library vs Vector Database by
    Weaviate*](https://weaviate.io/blog/vector-library-vs-vector-database) 
-   [*Not All Vector Databases Are Made Equal by Dmitry
    Kan*](https://towardsdatascience.com/milvus-pinecone-vespa-weaviate-vald-gsi-what-unites-these-buzz-words-and-what-makes-each-9c65a3bd0696)
-   [*Open Source Vector Database Comparison by
    Zilliz*](https://zilliz.com/comparison)
-   []{#anchor}[*Do you actually need a vector database? by Ethan
    Rosenthal*](https://www.ethanrosenthal.com/2023/04/10/nn-vs-ann/)

Module 3 Resources
==================

LLM Chains

-   [*LangChain*](https://docs.langchain.com/)
-   [*OpenAI ChatGPT
    Plugins*](https://platform.openai.com/docs/plugins/introduction)

LLM Agents

-   [*Transformers
    Agents*](https://huggingface.co/docs/transformers/transformers_agents)
-   [*AutoGPT*](https://github.com/Significant-Gravitas/Auto-GPT)
-   [*Baby AGI*](https://github.com/yoheinakajima/babyagi)
-   [*Dust.tt*](https://dust.tt/)

Multi-stage Reasoning in LLMs

-   [*CoT Paradigms*](https://matt-rickard.com/chain-of-thought-in-llms)
-   [*ReAct Paper*](https://react-lm.github.io/)
-   [*Demonstrate-Search-Predict
    Framework*](https://github.com/stanfordnlp/dsp)

Module 4 Resources
==================

Fine-tuned models

-   [*HF
    leaderboard*](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 
-   [*MPT-7B*](https://www.mosaicml.com/blog/mpt-7b)
-   [*Stanford
    Alpaca*](https://crfm.stanford.edu/2023/03/13/alpaca.html)
-   [*Vicuna*](https://lmsys.org/blog/2023-03-30-vicuna/) 
-   [*DeepSpeed on
    Databricks*](https://www.databricks.com/blog/2023/03/20/fine-tuning-large-language-models-hugging-face-and-deepspeed.html)

Databricks' Dolly

-   [*Dolly v1
    blog*](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html)
-   [*Dolly v2
    blog*](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
-   [*Dolly on Hugging
    Face*](https://huggingface.co/databricks/dolly-v2-12b)
-   [*Build your own
    Dolly*](https://www.databricks.com/resources/webinar/build-your-own-large-language-model-dolly)

Evaluation and Alignment in LLMs

-   [*HONEST*](https://huggingface.co/spaces/evaluate-measurement/honest)
-   [*LangChain
    Evaluate*](https://docs.langchain.com/docs/use-cases/evaluation)
-   [*OpenAI's post on InstructGPT and
    Alignment*](https://openai.com/research/instruction-following)
-   [*Anthropic AI Alignment
    Papers*](https://www.anthropic.com/index?subjects=alignment)

Module 5 Resources
==================

Social Risks and Benefits of LLMs

-   [*Weidinger et al 2021
    (DeepMind)*](https://arxiv.org/pdf/2112.04359.pdf)
-   [*Bender et al
    2021*](https://dl.acm.org/doi/10.1145/3442188.3445922)
-   [*Mokander et al
    2023*](https://link.springer.com/article/10.1007/s43681-023-00289-2)
-   [*Rillig et al
    2023*](https://pubs.acs.org/doi/pdf/10.1021/acs.est.3c01106)
-   [*Pan et al 2023*](https://arxiv.org/pdf/2305.13661.pdf)

Hallucination

-   [*Ji et al 2022*](https://arxiv.org/pdf/2202.03629.pdf)

Bias evaluation metrics and tools

-   [*NeMo Guardrails*](https://github.com/NVIDIA/NeMo-Guardrails)
-   [*Guardrails.ai*](https://shreyar.github.io/guardrails/)
-   [*Liang et al 2022*](https://arxiv.org/pdf/2211.09110.pdf)

Other general reading

-   [*All the Hard Stuff Nobody Talks About when Building Products with
    LLMs by
    Honeycomb*](https://www.honeycomb.io/blog/hard-stuff-nobody-talks-about-llm)    
-   [*Science in the age of large language models by Nature Reviews
    Physics*](https://www.nature.com/articles/s42254-023-00581-4)
-   [*Language models might be able to self-correct biases---if you ask
    them by MIT Technology
    Review*](https://www.technologyreview.com/2023/03/20/1070067/language-models-may-be-able-to-self-correct-biases-if-you-ask-them-to/)

****Module 6 Resources****
==========================

****

****General MLOps****

-   [*"*](https://www.databricks.com/resources/ebook/the-big-book-of-mlops)[*The
    Big Book of
    MLOps"*](https://www.databricks.com/resources/ebook/the-big-book-of-mlops)
    (eBook overviewing MLOps)

    -   Blog post (short) version:
        [*"*](https://www.databricks.com/blog/2022/06/22/architecting-mlops-on-the-lakehouse.html)[*Architecting
        MLOps on the
        Lakehouse"*](https://www.databricks.com/blog/2022/06/22/architecting-mlops-on-the-lakehouse.html)
    -   MLOps in the context of Databricks documentation
        ([*AWS*](https://docs.databricks.com/machine-learning/mlops/mlops-workflow.html),
        [*Azure*](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/mlops/mlops-workflow),
        [*GCP*](https://docs.gcp.databricks.com/machine-learning/mlops/mlops-workflow.html))

****LLMOps****

-   Blog post: Chip Huyen on "[*Building LLM applications for
    production*](https://huyenchip.com/2023/04/11/llm-engineering.html)"

[*****MLflow*****](https://mlflow.org/)

-   [*Documentation*](https://mlflow.org/docs/latest/index.html)

    -   [*Quickstart*](https://mlflow.org/docs/latest/quickstart.html)
    -   [*Tutorials and
        examples*](https://mlflow.org/docs/latest/tutorials-and-examples/index.html)
    -   Overview in Databricks
        ([*AWS*](https://docs.databricks.com/mlflow/index.html),
        [*Azure*](https://learn.microsoft.com/en-us/azure/databricks/mlflow/),
        [*GCP*](https://docs.gcp.databricks.com/mlflow/index.html))

[*****Apache Spark*****](https://spark.apache.org/)

-   [*Documentation*](https://spark.apache.org/docs/latest/index.html)

    -   [*Quickstart*](https://spark.apache.org/docs/latest/quick-start.html)

-   Overview in Databricks
    ([*AWS*](https://docs.databricks.com/spark/index.html),
    [*Azure*](https://learn.microsoft.com/en-us/azure/databricks/spark/),
    [*GCP*](https://docs.gcp.databricks.com/spark/index.html))

[*****Delta Lake*****](https://delta.io/)

-   [*Documentation*](https://docs.delta.io/latest/index.html)
-   Overview in Databricks
    ([*AWS*](https://docs.databricks.com/delta/index.html),
    [*Azure*](https://learn.microsoft.com/en-us/azure/databricks/delta/),
    [*GCP*](https://docs.gcp.databricks.com/delta/index.html))
-   [*Lakehouse Architecture (CIDR
    paper)*](https://www.cidrdb.org/cidr2021/papers/cidr2021_paper17.pdf)
