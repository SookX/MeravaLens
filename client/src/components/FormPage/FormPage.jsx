import { Box, Grid } from "@mui/material"
import satellite from "../../img/satellite.webp"
import { ImageBox, Section } from "./styling"


const FormPage = ({ children }) => {

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