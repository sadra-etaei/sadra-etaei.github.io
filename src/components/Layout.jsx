import { useEffect } from 'react'
import { NavLink, Outlet, Link, useLocation } from 'react-router-dom'
import ThemeToggle from './ThemeToggle.jsx'
import { site } from '../config.js'

export default function Layout() {
  const { pathname } = useLocation()

  useEffect(() => {
    window.scrollTo(0, 0)
  }, [pathname])

  return (
    <div className="shell">
      <header className="site-header">
        <Link to="/" className="wordmark" aria-label="Home">
          <span className="wordmark-dot" aria-hidden="true" />
          {site.name}
        </Link>
        <nav className="site-nav">
          <NavLink to="/" end>
            home
          </NavLink>
          <NavLink to="/posts">posts</NavLink>
          <NavLink to="/projects">projects</NavLink>
          <ThemeToggle />
        </nav>
      </header>

      <main className="site-main">
        <Outlet />
      </main>

      <footer className="site-footer">
        <span>
          © {new Date().getFullYear()} {site.name}
        </span>
        <span className="footer-links">
          {site.links.map((l) => (
            <a key={l.label} href={l.url} target="_blank" rel="noreferrer">
              {l.label}
            </a>
          ))}
        </span>
      </footer>
    </div>
  )
}
