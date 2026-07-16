import ProjectList from '../components/ProjectList.jsx'
import { projects, site } from '../config.js'

export default function Projects() {
  const github = site.links.find((l) => l.label === 'GitHub')

  return (
    <>
      <h1 className="page-title">Projects</h1>
      <p className="page-lede">
        A selection of things I've built. More on{' '}
        <a href={github?.url ?? 'https://github.com'} target="_blank" rel="noreferrer">
          GitHub
        </a>
        .
      </p>
      <ProjectList projects={projects} />
    </>
  )
}
