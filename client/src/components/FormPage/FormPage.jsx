import { Box, Card, Grid, Stack, styled, Typography } from "@mui/material"
import { theme } from "../../theme/theme"
import satellite from "../../img/satellite.webp"


const FormPage = ({ children }) => {
    const Section = styled(Grid)(({ theme })=>({
        minHeight: "100vh"
    }))


    const Image = styled(Box)({
        height: "100%",
        backgroundImage: `url(${satellite})`,
        backgroundSize: "cover"
    })
    

    return (
        <Section container>
            <Grid size={5}>
                <Image />
            </Grid>
            <Grid size="grow">
                { children }
            </Grid>
        </Section>
    )
}

export default FormPage