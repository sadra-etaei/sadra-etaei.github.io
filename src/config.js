// ─── Edit this file to make the site yours ───────────────────────────────
// Everything personal lives here: name, bio, links, projects.

export const site = {
  name: 'Sadra Etaei',
  // Short one-liner shown under your name on the home page.
  tagline: 'AI researcher',
  // A couple of sentences about you, shown on the home page.
  intro: `I like math , nlp , RL and I like writing about the cool things I build`,
  // Longer bio for the About section on the home page.
  about: `I'm an AI researcher with a focus on mathematics, natural language processing, and reinforcement learning. I like building models from first principles — transformers, language models, word embeddings — and writing about what I learn along the way.

When I'm not coding, you'll find me reading papers, experimenting with new architectures, or turning a tricky idea into something that actually runs.`,
  // Place your resume at public/resume.pdf (or change the filename here).
  resumeFile: 'resume.pdf',
  email: 'etaeisadra@gmail.com',
  links: [
    { label: 'GitHub', url: 'https://github.com/sadra-etaei' },
    { label: 'X', url: 'https://x.com/EtaeiSadra' },
    { label: 'LinkedIn', url: 'https://linkedin.com/in/yourusername' },
  ],
}

export const projects = [
  {
    title: 'img-transformer',
    description:
      'a reimplementation of the image transformer paper , implemented 2D local Attention and a vision transformer',
    url: 'https://github.com/sadra-etaei/img-transformer',
    tags: ['NLP', 'Image Generation'],
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
