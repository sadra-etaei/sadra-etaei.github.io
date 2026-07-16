import PostList from '../components/PostList.jsx'
import { posts } from '../lib/posts.js'

export default function Posts() {
  // Group posts by year, newest first.
  const byYear = posts.reduce((acc, post) => {
    const year = post.date.slice(0, 4)
    ;(acc[year] ||= []).push(post)
    return acc
  }, {})
  const years = Object.keys(byYear).sort((a, b) => b - a)

  return (
    <>
      <h1 className="page-title">Posts</h1>
      <p className="page-lede">
        Notes, essays and things I've learned. {posts.length}{' '}
        {posts.length === 1 ? 'post' : 'posts'} so far.
      </p>
      {years.map((year) => (
        <section key={year} className="year-group">
          <h2 className="year-label">{year}</h2>
          <PostList posts={byYear[year]} />
        </section>
      ))}
    </>
  )
}
