import { Box, Grid, styled } from "@mui/material"
import { theme } from "../../theme/theme"
import satellite from "../../img/satellite.webp"


const FormPage = ({ children }) => {
    const Section = styled(Grid)(({ theme })=>({
        height: "100vh"
    }))


    const Image = styled(Box)({
        height: "100vh",
        backgroundImage: `url(${satellite})`,
        backgroundSize: "cover",
        backgroundPosition: "right"
    })
    

    return (
        <Section container>
            <Grid size={{ md: 4, lg: 5 }}>
                <Image />
            </Grid>
            <Grid size="grow">
                <Box sx={{ height: "100vh", overflow: "auto" }}>
                    { children }
                </Box>
            </Grid>
        </Section>
    )
}

export default FormPage