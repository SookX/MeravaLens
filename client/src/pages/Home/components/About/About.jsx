import { Box, Grid, Typography } from "@mui/material"
import sat from "../../../../img/sat-example.webp"
import segm from "../../../../img/sat-segm-example.webp"
import { ImageBox, ImageContainer, Section } from "./styling"

const About = () => {



    return (
        <Section id="about" spacing={{ xs: 4, lg: 8 }} container>
            <Grid size={{ xs: 12, lg: 7 }}>
                <ImageContainer>
                    <ImageBox><img className="about-img sat" src={sat} alt="Satellite shot" /></ImageBox>
                    <ImageBox sx={{ right: "64px", bottom: "20px", transform: { xs: "none", lg: "rotate(20deg)" } }}><img className="about-img segm" src={segm} alt="Segmented satellite shot" /></ImageBox>
                </ImageContainer>
            </Grid>
            <Grid size="grow">
                <Box mb={4}>
                    <Typography variant="h2" color="primary">About</Typography>
                    <Typography mb={1} variant="body1">Every day, satellites capture terabytes of imagery covering every inch of Earth’s surface — but turning that raw data into meaningful, actionable insights remains a challenge. We built <Typography color="primary" variant="span">MeravaLens</Typography> to change that. </Typography>
                    <Typography variant="body1">It is an AI-powered platform that transforms satellite imagery into rich, structured geospatial insights.</Typography>
                </Box>

                <Typography variant="h4" color="primary">The model</Typography>
                <Typography mb={1} variant="body1">MeravaLens integrates a custom-built <Typography variant="span" color="primary">ResUNet</Typography> deep learning model to perform high-resolution <Typography variant="span" color="primary">semantic segmentation</Typography> on satellite imagery. This model enables detailed pixel-level classification of land cover types such as buildings, roads, forests, agriculture, and more.</Typography>
                <Typography variant="body1">The model is trained and evaluated on the <Typography variant="span" color="primary">LoveDA (Land-cOver Domain Adaptation)</Typography> dataset, which is a comprehensive benchmark tailored for semantic segmentation in remote sensing. Classification head for urban vs rural domain prediction</Typography>
            </Grid>
        </Section>
    )
}

export default About