import { Typography, Grid, Stack, Button } from '@mui/material'
import EarthCanva from './components/EarthScene/EarthScene'
import { Link } from 'react-router-dom'
import { useContext, useEffect } from 'react'
import { DataContext } from '../../../../context/DataContext'
import { HashLink } from "react-router-hash-link"
import { Circle, LeftDivider, RightDivider, TextBox, Welcome } from './styling'



const Hero = () => {
    const { setLoading } = useContext(DataContext)

    useEffect(() => {
        setLoading(true)
    }, [])



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

                    <Stack gap={1} direction="row" sx={{ alignSelf: { xs: "center", md: "start" } }}>
                        <Button color="text" variant="text"><HashLink to="/#about" style={{ color: "inherit" }}>Learn more</HashLink></Button>
                        <Button variant="outlined"><Link to="/register" style={{ color: "inherit" }}>Get started</Link></Button>
                    </Stack>

                </TextBox>
            </Grid>
            <Grid size={{ xs: 12, md: "grow" }}><EarthCanva /></Grid>
        </Grid>
    )
}

export default Hero