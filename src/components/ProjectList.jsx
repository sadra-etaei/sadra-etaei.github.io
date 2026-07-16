export default function ProjectList({ projects }) {
  return (
    <ul className="project-list">
      {projects.map((p) => (
        <li key={p.title} className="project-card">
          <div className="project-head">
            <a href={p.url} target="_blank" rel="noreferrer" className="project-title">
              {p.title}
            </a>
            <span className="project-year">{p.year}</span>
          </div>
          <p className="project-description">{p.description}</p>
          <div className="project-tags">
            {p.tags.map((tag) => (
              <span key={tag} className="tag">
                {tag}
              </span>
            ))}
          </div>
        </li>
      ))}
    </ul>
  )
}
