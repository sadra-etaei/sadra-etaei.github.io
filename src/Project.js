import cat from "./images/github-logo.png"
export default function Project(props) {
    return (
        <div className="poject">
            <img className="pro-img" src={props.src} alt=" " />
            <h2>{props.name}</h2>
            <div className="github">
                <a href={props.link}><img className="github-logo" src={cat} alt="" /></a>
            </div>
        </div>
    )
}