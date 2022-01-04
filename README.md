# Abstractive Text Summarization of Lecture Video Transcripts


The expansion of the internet has resulted in an abundance of multi-sourced information being created in the form of videos available for public consumption in recent years. This includes the big supply of online educational videos varying in duration, content, and presentation style. Lecture videos usually have a long duration (more than 30 minutes), making it hard to grasp the main ideas or to choose suitable videos to watch without summary of the videos and thus there is a necessity for developing such summarization techniques. The problem of creating a summary of lecture videos has been less studied. While the available textual summarization methods are trained and built for written texts such as news and blog articles, this mismatch of data sources may lead to issues while making use of these techniques to spoken language as in videos.

This project is intended to summarize educational videos by experimenting with existing methods of long-text summarization for video transcript summarization. We used texts scraped from the Open Yale Courses website (https://oyc.yale.edu/courses); the corpus contains more than 1000 long transcripts and corresponding summaries from lectures of 41 courses. Our implementation takes over from fine-tuning BART model and compares performance of abstractive summarization alone and that of a combined approach of both extractive and abstractive summarization. 

Please find the [PDF](https://github.com/siwei-li/NLP_summarization/blob/master/Report.pdf) report in the repo.

---

Using Hugging Face models and pipeline

https://huggingface.co/transformers/notebooks.html

Sample notebooks:

https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb

https://github.com/ohmeow/ohmeow_website/blob/master/_notebooks/2020-05-23-text-generation-with-blurr.ipynb



---

Please see

https://github.com/alebryvas/berk266 for abstractive summarization which alters BertSum’s code

 https://github.com/dmmiller612/lecture-summarizer for unsupervised summarization


