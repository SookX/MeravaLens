import { Link, Typography } from "@mui/material"
import logo from "../../img/logo.webp"
import devpost from "../../img/devpost.svg"
import github from "../../img/github.svg"
import { theme } from "../../theme/theme"
import { StyledFooter, IconContainer, StyledDivider } from "./styling"

const Footer = () => {

    return (
        <StyledFooter>
            <img className="footer-logo" src={logo} alt="Merava Lens Logo" />

            <Typography mt={1} mb={2} variant="body1">This project was submitted to the <Link target="_blank" href='https://launchhacks-iv.devpost.com/?_gl=1*dx3rw7*_ga*MTc5NTg5NzMzLjE3NTEwMjk0MzE.'>LaunchHacks IV</Link> hackathon.</Typography>

            <IconContainer>
                <StyledDivider sx={{ height: theme.spacing(6) }} />
                <Link href="https://devpost.com/software/meravalens" target="_blank"><img className="footer-icon" src={devpost} alt="Devpost Logo" /></Link>
                <StyledDivider />
                <Link href="https://github.com/SookX/MeravaLens" target="_blank"><img className="footer-icon" src={github} alt="Devpost Logo" /></Link>
            </IconContainer>

            <Typography variant="body2" color="text.secondary">&copy; Copyright {new Date().getFullYear()}</Typography>
        </StyledFooter>
    )
}

export default Footer