import { Box, Typography, Grid, styled, Button, Divider } from '@mui/material'
import EarthCanva from './components/EarthScene/EarthScene'
import { theme } from '../../theme/theme'
import { Link } from 'react-router-dom'

const Home = () => {
    const TextBox = styled(Box)(({ theme })=>({
        padding: `${theme.spacing(16)} ${theme.spacing(16)}`,
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        [theme.breakpoints.down("md")]: {
            alignItems: "center",
            textAlign: "center",
            padding: `${theme.spacing(16)} ${theme.spacing(8)}`,
            paddingBottom: 0
        },
        [theme.breakpoints.down("sm")]: {
            padding: `${theme.spacing(16)} ${theme.spacing(4)}`,
            paddingBottom: 0
        },
        height: "100%",
        position: "relative",
        overflow: "hidden",
    }))

    const Welcome = styled(Box)(({ theme })=>({
        position: "relative"
    }))

    const StyledDivider = styled(Divider)(({ theme })=>({
        position: "absolute",
        width: theme.spacing(8),
        top: "50%",
        transfrom: "translateY(-50%)",
        backgroundColor: theme.palette.text.default
    }))

    const LeftDivider = styled(StyledDivider)(({ theme })=>({
        left: `-${theme.spacing(9)}`,
    }))

    const RightDivider = styled(StyledDivider)(({ theme })=>({
        [theme.breakpoints.up("md")]: {
            display: "none"
        },
        right: theme.spacing(-9)
    }))

    const Circle = styled(Box)(({ theme })=>({
        width: theme.spacing(80),
        aspectRatio: "1 / 1",
        border: `solid 1px ${theme.palette.text.dark}`,
        borderRadius: "100%",
        position: "absolute",
        left: "-30%",
        top: "50%",
        zIndex: 0,

        [theme.breakpoints.down("md")]: { display: "none" }
    }))

    return (
        <Grid container direction="row">
            <Grid size={{ xs: 12, md: 9, lg: 7 }}>
                <TextBox>
                    <Circle />
                    <Welcome>
                        <LeftDivider />
                        <Typography variant="h4" textTransform="uppercase">Welcome to</Typography>
                        <RightDivider />
                    </Welcome>

                    <Typography color="primary" sx={{ zIndex: "1" }} variant="h1">Merava Lens</Typography>

                    <Typography variant="body1" sx={{ zIndex: "1" }} mb={1}>Your Gateway to Real-Time Environmental Intelligence</Typography>
                    <Typography variant="body1" sx={{ zIndex: "1" }} mb={6}>MeravaLens is a next-generation satellite platform that brings together the power of multiple APIs and advanced  AI models to provide comprehensive, real-time environmental data – all in one place.</Typography>

                    <Button variant="outlined" sx={{ alignSelf: { xs: "center", md: "start" } }}><Link to="/register" style={{ color: "inherit" }}>Get started</Link></Button>
                </TextBox>
            </Grid>
            <Grid size={{ xs: 12, md: "grow" }}><EarthCanva /></Grid>
        </Grid>
    )
}

export default Home