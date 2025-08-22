import { Box, Grid, styled } from "@mui/material"
import satellite from "../../img/satellite.webp"


const FormPage = ({ children }) => {
    const Section = styled(Grid)(({ theme })=>({
        minHeight: "100vh"
    }))


    const ImageBox = styled(Box)({
        height: "100vh",
        width: "100%",
        position: "sticky",
        top: 0
    })
    

    return (
        <Section container>
            <Grid size={{ xs: 0, md: 4, lg: 5 }}>
                <ImageBox>
                    <img src={satellite} alt="Satellite" className="form-img" />
                </ImageBox>
            </Grid>
            <Grid size="grow">
                <Box sx={{  }}>
                    { children }
                </Box>
            </Grid>
        </Section>
    )
}

export default FormPage