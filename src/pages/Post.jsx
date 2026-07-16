import { useParams, Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeRaw from 'rehype-raw'
import 'katex/dist/katex.min.css'
import { getPost, formatDate } from '../lib/posts.js'
import NotFound from './NotFound.jsx'

export default function Post() {
  const { slug } = useParams()
  const post = getPost(slug)

  if (!post) return <NotFound />

  return (
    <article className="post">
      <header className="post-header">
        <h1 className="post-title">{post.title}</h1>
        <div className="post-meta">
          <time dateTime={post.date}>{formatDate(post.date)}</time>
          <span aria-hidden="true">·</span>
          <span>{post.readingTime} min read</span>
          {post.tags.length > 0 && (
            <>
              <span aria-hidden="true">·</span>
              <span className="post-tags">
                {post.tags.map((tag) => (
                  <span key={tag} className="tag">
                    {tag}
                  </span>
                ))}
              </span>
            </>
          )}
        </div>
      </header>

      <div className="prose">
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeRaw, rehypeKatex]}
        >
          {post.content}
        </ReactMarkdown>
      </div>

      <footer className="post-footer">
        <Link to="/posts">← Back to all posts</Link>
      </footer>
    </article>
  )
}
