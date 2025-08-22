import { useContext } from "react"
import { DashboardContext } from "../../../../Dashboard"
import { Box, IconButton, Stack, Tooltip, Typography } from "@mui/material"
import { Icon, StyledCard } from "./styling"

const SummaryCard = () => {
    // Gets dashboard data
    const { summary } = useContext(DashboardContext)


    
    return (
        <StyledCard>
            <Box mb={2}>
                {
                    summary.map((component, i) => (
                        <Box key={i} mb={1}>
                            <Typography variant={`${i == 0 ? "h3" : "h5"}`} color="primary">{component.title}</Typography>
                            {
                                component.body ?
                                <Typography variant="body1">{component.body}</Typography>
                                :
                                component.paragraphs.map((paragraph, j) => (
                                    <Stack direction="row" gap={1}>
                                        <Typography variant="body1"><Typography variant="span" color="primary.dark">{paragraph.title}: </Typography> {paragraph.body}</Typography>
                                    </Stack>
                                ))
                            }
                        </Box>
                    ))
                }
            </Box>

            <Tooltip title="Generated with llama3.3" placement="right">
                <IconButton><Icon color="primary" /></IconButton>
            </Tooltip>
        </StyledCard>
    )
}

export default SummaryCard