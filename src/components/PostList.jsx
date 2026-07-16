import { Link } from 'react-router-dom'
import { formatDate } from '../lib/posts.js'

export default function PostList({ posts }) {
  return (
    <ul className="post-list">
      {posts.map((post) => (
        <li key={post.slug} className="post-list-item">
          <Link to={`/posts/${post.slug}`} className="post-list-link">
            <span className="post-list-title">{post.title}</span>
            <span className="post-list-date">{formatDate(post.date)}</span>
          </Link>
          {post.summary && <p className="post-list-summary">{post.summary}</p>}
        </li>
      ))}
    </ul>
  )
}
