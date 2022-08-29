import './styles.css';
import img1 from "./images/IMG_1522.jpeg"
import Skill from './Skill';
import React from 'react';
import my_skills from "./my_skills.json"
import Project from './Project';
import data from "./my_projects";
import mouse from "./images/mouse-cursor.png"
import article from "./images/article.png"
import gmail from "./images/gmail.png"
import phone from "./images/apple.png"
import insta from "./images/instagram.png"
import twitter from "./images/twitter.png"


function App() {
  const [skills, setSkills] = React.useState([])
  const [projects, setProjects] = React.useState([])
  const [mode, setMode] = React.useState("light")
  const [lan, setLan] = React.useState("en")
  function get() {
    my_skills.map(s => setSkills(prev => [...prev, <Skill name={s.name} proficiency={s.pro} />]))
    data.map(pro => setProjects(prev => [...prev, <Project name={pro.name} src={pro.src} />]))
  }
  function changeMode() {
    setMode(prevMode => prevMode === "light" ? "dark" : "light")
  }
  function changeLan() {
    setLan(prevMode => prevMode === "en" ? "fa" : "en")
  }
  React.useEffect(get, [])
  const style1 = {
    color: mode === "light" ? "black" : "white",
    backgroundColor: mode === "light" ? "white" : "rgb(5, 33, 45)  ",
    display: lan === "en" ? "flex" : "none"
  }
  const style2 = {
    color: mode === "light" ? "black" : "white",
    // backgroundColor: mode === "light mode" ? "rgb(13, 6, 55)" : "white"
  }
  const style3 = {
    color: mode === "light" ? "black" : "white",
    backgroundColor: mode === "light" ? "white" : "rgb(13, 6, 55)",
    display: lan === "en" ? "none" : "flex"
  }
  const style4 = {
    direction: lan === "en" ? "ltr" : "rtl",
    flexDirection: lan === "en" ? "row" : "row-reverse",
    marginLeft: lan === "en" ? "100px" : "10px"
  }
  return (
    <main>
      <div className='en' style={style1} >
        <div className='first-sec'>
          <div id='en-nav' style={style4} className='navbar'>
            <input className='lan' onClick={changeLan} type={"button"} value={lan} />

            <div className='anchors'>
              <a href='#about'>About</a>
              <a href='#skills'>Skills</a>
              <a href='#projects'>Projects</a>
              <a href='#articles'>Articles</a>
              <a href='#contact'>Contact</a>
            </div>

            <input className='mode' onClick={changeMode} type={"button"} value={mode} />

          </div>
          <div className='intro'>
            <h1>Sadra Etaei</h1>
            <h2>full stack web developer</h2>
            <img id="mouse" src={mouse} alt="" />
          </div>
        </div>
        <div id='about' className='about'>
          <div className='info'>
            <h1 >About Me </h1>
            <p> I am a 16 year old iranian  web developer , AI inthusiast , photographer and translator based in Urmia / Iran
              I first started programming when I was 13 and I`ve been a web developer for a year now ,
              I`m also fluent in english and persian .</p>
          </div>
          <img id='personal' src={img1} alt="" />
        </div>
        <div id='skills' className='skills'>
          <h1>Skills</h1>
          <div style={style2} className='box'>
            <div className='first-col'>
              {skills.slice(0, 5)}
            </div>
            <div className='sec-col'>
              {skills.slice(5, 10)}
            </div>
          </div>
        </div>
        <div className='lans'>
          <h1>Languages</h1>
          <div className='box-3'>
            <div className='lan'>
              <div className='per'>
                <h2>English :</h2>
                <h3>90%</h3>
              </div>

              <div className='bar-1'>
                <div className='color-2'></div>
              </div>
            </div>
            <div className='lan-2'>
              <div className='per'>
                <h2>Persian :</h2>
                <h3>100%</h3>
              </div>
              <div className='bar-1'>
                <div className='color-1'></div>
              </div>
            </div>
          </div>
        </div>
        <div id='projects' className='projects'>
          <h1>Projects</h1>
          <div className='box-2'>
            {projects.slice(0, 4)}
          </div>
        </div>
        <div id='articles' className='articles'>
          <h1>Articles</h1>
          <p>There are no articles right now</p>
          <div className='article-logo-parent'>
            <img id='article-logo' src={article} alt="" />
          </div>
        </div>
        <div id='contact' className='contact'>
          <h1>Contact Me</h1>
          <p>you can either  contact me through social media or in person </p>
          <div className='contact-box'>
            <div className='social'>
              <a href='https://mail.google.com/'><img src={gmail} alt="" /></a>
              <p>etaeisadra@gmail.com</p>
            </div>
            <div className='social'>
              <a href='a'><img src={phone} alt="" /></a>
              <p>09021708686</p>
            </div>
            <div className='social'>
              <a href='https://www.instagram.com/sadra.etaei/'><img src={insta} alt="" /></a>
              <p>sadra.etaei</p>
            </div>
            <div className='social'>
              <a href='https://twitter.com/EtaeiSadra'><img src={twitter} alt="" /></a>
              <p>EtaeiSadra</p>
            </div>
          </div>
        </div>
        <div className='footer'>
          {/* <h1>Sadra Etaei</h1> */}
          <p>© 2022 Sadra Etaei</p>
        </div>
      </div >
      <div lang="fa-IR" className="fa" style={style3} >
        <div className='first-sec'>
          <div className='up-sec'>
            <div id='fa-nav' className='navbar'>
              <input className='lan' onClick={changeLan} type={"button"} value={lan} />
              <div className='anchors'>
                <a href='#about2'>درباره من</a>
                <a href='#skills2'>مهارت ها</a>
                <a href='#projects2'>پروژه ها</a>
                <a href='#articles2'>مقالات</a>
                <a href='#contact2'>ارتباط با من</a>
              </div>
              <input className='mode' onClick={changeMode} type={"button"} value={mode} />
            </div>
          </div>
          <div className='intro'>
            <h1>صدرا اعطایی </h1>
            <h2>توسعه دهنده وب</h2>
            <img id="mouse" src={mouse} alt="" />
          </div>
        </div>
        <div id='about2' className='about'>
          <div className='info'>
            <h1>درباره من </h1>
            <p>  من صدرا اعطایی برنامه نویس , توسعه دهنده وب , عکاس و مترجم متولد 1385 هستم و چند سالی هست که کد نویسی می کنم و الان یک سال هست که به صورت نخصصی برنامه نویس وب می کنم</p>
          </div>
          <img id='personal' src={img1} alt="" />
        </div>
        <div id='skills2' className='skills'>
          <h1>مهارت ها</h1>
          <div style={style2} className='box'>
            <div className='first-col'>
              {skills.slice(0, 5)}
            </div>
            <div className='sec-col'>
              {skills.slice(5, 10)}
            </div>
          </div>
        </div>
        <div className='lans'>
          <h1>زبان ها</h1>
          <div className='box-3'>
            <div className='lan'>
              <div className='per'>
                <h2  >انگلیسی   :   </h2>
                <h3>90%</h3>
              </div>

              <div className='bar-1'>
                <div className='color-2'></div>
              </div>
            </div>
            <div className='lan-2'>
              <div className='per'>
                <h2>فارسی     :</h2>
                <h3>100%</h3>
              </div>
              <div className='bar-1'>
                <div className='color-1'></div>
              </div>
            </div>
          </div>
        </div>
        <div id='projects2' className='projects'>
          <h1>پروژه ها</h1>
          <div className='box-2'>
            {projects.slice(0, 4)}
          </div>
        </div>
        <div id='articles2' className='articles'>
          <h1>مقالات</h1>
          <p>هنوز مقاله ای قرار نگرفته است</p>
          <div className='article-logo-parent'>
            <img id='article-logo' src={article} alt="" />
          </div>
        </div>
        <div id='contact2' className='contact'>
          <h1>ارتباط با من</h1>
          <p>یرای ارتباط با من می توانید به صورت حضوری ویا از طریق فضای مجازی اقدام کنید</p>
          <div className='contact-box'>
            <div className='social'>
              <a href='https://mail.google.com/'><img src={gmail} alt="" /></a>
              <p>etaeisadra@gmail.com</p>
            </div>
            <div className='social'>
              <a href='a'><img src={phone} alt="" /></a>
              <p>09021708686</p>
            </div>
            <div className='social'>
              <a href='https://www.instagram.com/sadra.etaei/'><img src={insta} alt="" /></a>
              <p>sadra.etaei</p>
            </div>
            <div className='social'>
              <a href='https://twitter.com/EtaeiSadra'><img src={twitter} alt="" /></a>
              <p>EtaeiSadra</p>
            </div>
          </div>
        </div>
        <div className='footer'>
          <p>© 2022 Sadra Etaei</p>
        </div>
      </div >
    </main >
  )
}

export default App;
