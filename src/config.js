// ─── Edit this file to make the site yours ───────────────────────────────
// Everything personal lives here: name, bio, links, projects.

export const site = {
  name: 'Sadra Etaei',
  // Short one-liner shown under your name on the home page.
  tagline: 'AI researcher',
  // A couple of sentences about you, shown on the home page.
  intro: 'I like math , nlp , RL and I`m also interested in mechanistic interpretability and AI safety ',
  // Longer bio for the About section on the home page.
  about: `I'm an Electrical engineering student studying in Shiraz university and an AI Researcher / Engineer  with a focus on  natural language processing, and reinforcement learning. I like building things from scratch  and writing about what I learn along the way.

I mostly like building intelligence and more than that I like seeing how that intelligence works `,
  // Place your resume at public/resume.pdf (or change the filename here).
  resumeFile: 'resume.pdf',
  email: 'etaeisadra@gmail.com',
  links: [
    { label: 'GitHub', url: 'https://github.com/sadra-etaei' },
    { label: 'X', url: 'https://x.com/EtaeiSadra' },
    { label: 'LinkedIn', url: 'https://www.linkedin.com/in/sadra-etaei-143862350/' },
  ],
}

export const projects = [
  // {
  //   title: 'arithmetic-circuits',
  //   description:
  //     "reverse-engineered how Qwen2.5-0.5B adds two numbers — the attention heads that fetch digits, the MLPs that compute, and the model's most-significant-first carry",
  //   url: 'https://github.com/sadra-etaei/arithmetic_circuits',
  //   tags: ['Interpretability', 'LLMs'],
  //   year: '2026',
  // },
  {
    title: 'BERT',
    description:
      'a reimplementation of BERT',
    url: 'https://github.com/sadra-etaei/BERT',
    tags: ['NLP', 'pretraining'],
    year: '2026',
  },
  {
    title: 'img-transformer',
    description:
      'a reimplementation of the image transformer paper , implemented 2D local Attention and a vision transformer',
    url: 'https://github.com/sadra-etaei/img-transformer',
    tags: ['NLP', 'Image Generation','ViT'],
    year: '2026',
  },
  {
    title: 'mini-transformer',
    description:
      'a reimplementation of the Attention is all you need paper using pytorch ',
    url: 'https://github.com/sadra-etaei/mini-transformer',
    tags: ['Transformer', 'NLP'],
    year: '2025',
  },
  {
    title: 'char-lm',
    description: 'built a character level language model using an N-Layer LSTM in python',
    url: 'https://github.com/sadra-etaei/char-lm',
    tags: ['NLP','Language-modeling'],
    year: '2026',
  },
  {
    title: 'Word2vec',
    description: 'implemented Word2vec from scratch using python and numpy',
    url: 'https://github.com/sadra-etaei/word2vec',
    tags: ['NLP','Word embeddings'],
    year: '2026',
  },
  {
    title: 'GloVe',
    description: 'implemented GloVe from scratch using python and numpy',
    url: 'https://github.com/sadra-etaei/GloVe',
    tags: ['NLP','Word embeddings'],
    year: '2026',
  },
]
