import { useContext } from "react"
import { DashboardContext } from "../../../../Dashboard"
import { Box, Card, IconButton, Stack, styled, Tooltip, Typography } from "@mui/material"
import { theme } from "../../../../../../theme/theme"
import { HelpOutline, QuestionMarkOutlined } from "@mui/icons-material"

const SummaryCard = () => {
    // Gets dashboard data
    const { summary } = useContext(DashboardContext)



    const StyledCard = styled(Card)(({ theme }) => ({
        padding: theme.spacing(3)
    }))



    const Icon = styled(HelpOutline)(({ theme })=>({
        color: theme.palette.primary.main,
        cursor: "pointer",
        width: theme.spacing(4),
        height: theme.spacing(4)
    }))


    
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