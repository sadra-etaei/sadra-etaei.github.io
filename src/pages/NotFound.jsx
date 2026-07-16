import { Link } from 'react-router-dom'

export default function NotFound() {
  return (
    <div className="notfound">
      <h1 className="page-title">404</h1>
      <p className="page-lede">This page doesn't exist (yet).</p>
      <Link to="/">← Take me home</Link>
    </div>
  )
}
