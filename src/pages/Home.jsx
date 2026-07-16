import { Link } from 'react-router-dom'
import PostList from '../components/PostList.jsx'
import ProjectList from '../components/ProjectList.jsx'
import { posts } from '../lib/posts.js'
import { projects, site } from '../config.js'

const resumeUrl = `${import.meta.env.BASE_URL}${site.resumeFile}`

export default function Home() {
  return (
    <>
      <section className="hero">
        <h1 className="hero-title">
          Hi, I'm {site.name.split(' ')[0]}
          <span className="accent">.</span>
        </h1>
        <p className="hero-tagline">{site.tagline}</p>
        <p className="hero-intro">{site.intro}</p>
        <p className="hero-contact">
          Find me on{' '}
          {site.links.map((l, i) => (
            <span key={l.label}>
              <a href={l.url} target="_blank" rel="noreferrer">
                {l.label}
              </a>
              {i < site.links.length - 2 ? ', ' : i === site.links.length - 2 ? ' or ' : ''}
            </span>
          ))}
          , or say hello at <a href={`mailto:${site.email}`}>{site.email}</a>.
        </p>
      </section>

      <section className="home-section">
        <div className="section-heading">
          <h2>About</h2>
        </div>
        <div className="about-content">
          <p className="about-text">{site.about}</p>
          <a href={resumeUrl} download className="resume-link">
            Download resume
          </a>
        </div>
      </section>

      <section className="home-section">
        <div className="section-heading">
          <h2>Recent projects</h2>
          <Link to="/projects" className="see-all">
            all projects →
          </Link>
        </div>
        <ProjectList projects={projects.slice(0, 2)} />
      </section>

      <section className="home-section">
        <div className="section-heading">
          <h2>Recent writing</h2>
          <Link to="/posts" className="see-all">
            all posts →
          </Link>
        </div>
        <PostList posts={posts.slice(0, 4)} />
      </section>
    </>
  )
}
