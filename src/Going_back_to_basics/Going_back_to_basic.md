GATs amplify variance twice. This is yet another reminder—despite years of working with deep learning—that even on *trivial synthetic binary tasks*, careless weight initialization can quietly ruin results. Initializing weights with a uniform (non–zero-mean) distribution noticeably degrades performance.

If this sounds obvious, run `1.py` and see it happen.
