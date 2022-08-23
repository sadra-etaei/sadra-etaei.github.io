export default function Skill(props) {
    const styles = {
        width : props.proficiency * 2
    }
    return (
        <div className='skill'>
            <div className='details'>
                <h3>{props.name} :</h3>
                <h4>{props.proficiency} % </h4>
            </div>
            <div className='bar'>
                <div style={styles} className='color'>

                </div>
            </div>
        </div>
    )
}